"""Build the human pairwise test (option A: 10 pairs, read the copy cards, 10 more pairs).

Pairs come from the newest 30% of each creator's reels (test split); the card statistics come only from the
oldest 70% (train split), so the cards cannot leak the answers. Writes the answer key and a JSON payload that the
test page embeds."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

OUT_DIR = config.PROCESSED_DIR
MIN_RATIO, MAX_DAYS, MIN_CHARS, PER_CREATOR = 2.0, 14, 60, 2
# Ads, coupons, help requests and party politics are not copy we want strangers to judge over WhatsApp.
EXCLUDE = re.compile(r"cupom|culpom|desconto|https?://|direct do insta|bolsonaro|lula|esquerda|direita|trump|tarcisio|nikolas|governador|eleic", re.I)
rng = np.random.default_rng(7)


def clean_caption(s: str, limit: int = 320) -> str:
    s = re.sub(r"(#\w+\s*)+$", "", (s or "").strip())          # trailing hashtag block
    s = re.sub(r"\s*\n\s*", " / ", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" /")
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "..."


def first_line(s: str) -> str:
    return (s or "").strip().split("\n")[0]


def card_stats(train: pd.DataFrame) -> dict:
    """Within-creator differences (viral minus non-viral share) for a few caption devices."""
    lab = train[train["y_viral"].notna()].copy()
    cap = lab["caption"].fillna("")
    fl = cap.map(first_line)
    feats = {
        "short_first_line": fl.str.len() <= 60,
        "how_to": cap.str.lower().str.contains(r"\bcomo\b"),
        "number_first_line": fl.str.contains(r"\d"),
        "money_number": lab["dev_money_number"].astype(bool),
        "second_person": lab["dev_second_person"].astype(bool),
        "cta_follow": lab["dev_cta_follow"].astype(bool),
    }
    out = {"n_creators": int(lab["username"].nunique())}
    for name, f in feats.items():
        diff = lab.assign(f=f.values).groupby("username").apply(
            lambda g: g.loc[g.y_viral == 1, "f"].mean() - g.loc[g.y_viral == 0, "f"].mean())
        out[name] = {"up": int((diff > 0.05).sum()), "down": int((diff < -0.05).sum()),
                     "mean_diff_pp": round(float(diff.mean()) * 100, 1)}
    return out


def build_pairs(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for user, g in test.groupby("username"):
        g = g.sort_values("posted_at").reset_index(drop=True)
        g["clean"] = g["caption"].map(clean_caption)
        g = g[(g["clean"].str.len() >= MIN_CHARS) & ~g["clean"].str.contains(EXCLUDE)].reset_index(drop=True)
        cands = []
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                gap = (g.posted_at[j] - g.posted_at[i]).total_seconds() / 86400
                if gap > MAX_DAYS:
                    break
                hi, lo = max(g.views[i], g.views[j]), max(min(g.views[i], g.views[j]), 1)
                if hi / lo >= MIN_RATIO:
                    both_hooks = bool(g.hook[i]) and bool(g.hook[j]) if "hook" in g else False
                    cands.append((both_hooks, hi / lo, i, j))
        cands.sort(key=lambda c: (c[0], min(c[1], 6)), reverse=True)
        used = set()
        for both_hooks, ratio, i, j in cands:
            if i in used or j in used:
                continue
            used.update((i, j))
            win, lose = (i, j) if g.views[i] > g.views[j] else (j, i)
            rows.append({"username": user, "sector": g.sector[i], "winner": g.reel_id[win], "loser": g.reel_id[lose],
                         "ratio": round(ratio, 2), "gap_days": round(abs(
                             (g.posted_at[i] - g.posted_at[j]).total_seconds()) / 86400, 1),
                         "winner_text": g.clean[win], "loser_text": g.clean[lose],
                         "winner_hook": (g.hook[win] or "") if both_hooks else "",
                         "loser_hook": (g.hook[lose] or "") if both_hooks else ""})
            if sum(r["username"] == user for r in rows) >= PER_CREATOR:
                break
    return pd.DataFrame(rows)


def main():
    d = pd.read_parquet(OUT_DIR / "dataset_v1.parquet")
    d["posted_at"] = pd.to_datetime(d["posted_at"], utc=True)
    d["hook"] = d["hook"].fillna("") if "hook" in d else ""
    stats = card_stats(d[d["split"] == "train"])
    pairs = build_pairs(d[d["split"] == "test"])
    pairs = pairs.sample(frac=1, random_state=3).reset_index(drop=True)
    # Two blocks with one pair per creator each where possible; winner side randomized.
    pairs["block"] = pairs.groupby("username").cumcount().map(lambda k: 1 if k == 0 else 2)
    singles = pairs.index[pairs.groupby("username")["username"].transform("size") == 1]
    pairs.loc[singles[1::2], "block"] = 2  # creators with one pair alternate, so both blocks are the same size
    pairs = pairs.sort_values("block", kind="stable").reset_index(drop=True)
    pairs["winner_side"] = rng.choice(["A", "B"], size=len(pairs))
    pairs["pair_id"] = [f"p{i + 1:02d}" for i in range(len(pairs))]
    pairs.to_csv(OUT_DIR / "human_test_key.csv", index=False, encoding="utf-8")
    items = []
    for r in pairs.itertuples():
        a_win = r.winner_side == "A"
        items.append({"id": r.pair_id, "block": int(r.block), "creator": r.username, "sector": r.sector,
                      "a": r.winner_text if a_win else r.loser_text, "b": r.loser_text if a_win else r.winner_text,
                      "ah": r.winner_hook if a_win else r.loser_hook, "bh": r.loser_hook if a_win else r.winner_hook,
                      "ans": r.winner_side})
    payload = {"pairs": items, "card_stats": stats}
    (OUT_DIR / "human_test_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(pairs.groupby(["block"]).size().to_dict(), pairs.username.value_counts().to_dict())
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
