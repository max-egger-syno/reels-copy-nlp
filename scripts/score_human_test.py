"""Score the human pairwise test from the result lines participants sent over WhatsApp.

Paste every message (any amount of surrounding chat text is fine) into data/human_test_responses.txt, then run
  python scripts/score_human_test.py
Writes results/human_test.json and results/human_test_answers.csv."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

RESP = ROOT / "data" / "human_test_responses.txt"
KEY = config.PROCESSED_DIR / "human_test_key.csv"
LINE = re.compile(r"TESTE-REELS v1 \|(?P<body>[^\n]+)")


def parse(text: str) -> pd.DataFrame:
    rows = []
    for n, m in enumerate(LINE.finditer(text)):
        f = dict(part.strip().split(":", 1) for part in m.group("body").split("|") if ":" in part)
        for phase in ("antes", "depois"):
            for pid, side in re.findall(r"(p\d{2})([AB])", f.get(phase, "")):
                rows.append({"participant": n, "name": f.get("nome", "-").strip(), "creates": f.get("cria", "").strip(),
                             "order": f.get("ordem", "").strip(), "phase": phase, "pair_id": pid, "choice": side,
                             "helped": int(f.get("ajudou", "0").strip() or 0), "minutes": f.get("min", "").strip(),
                             "comment": f.get("obs", "").strip()})
    return pd.DataFrame(rows)


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    if not RESP.exists():
        print(f"no responses yet: create {RESP}")
        return
    ans = parse(RESP.read_text(encoding="utf-8"))
    if ans.empty:
        print("no result lines found")
        return
    # The same person may send twice: keep the last message per name when a name was given.
    named = ans[ans.name != "-"]
    last = named.groupby("name").participant.max()
    ans = ans[(ans.name == "-") | ans.participant.isin(last.values)]
    key = pd.read_csv(KEY)[["pair_id", "winner_side", "username", "sector", "ratio"]]
    ans = ans.merge(key, on="pair_id", how="left")
    ans["correct"] = (ans.choice == ans.winner_side).astype(int)
    per = ans.groupby(["participant", "phase"]).correct.mean().unstack()
    out = {"n_participants": int(ans.participant.nunique()),
           "accuracy_all": round(float(ans.correct.mean()), 3),
           "accuracy_before_cards": round(float(ans[ans.phase == "antes"].correct.mean()), 3),
           "accuracy_after_cards": round(float(ans[ans.phase == "depois"].correct.mean()), 3),
           "mean_gain_pp": round(float((per["depois"] - per["antes"]).mean() * 100), 1),
           "gain_ci95_pp": [round(v * 100, 1) for v in boot_ci((per["depois"] - per["antes"]).to_numpy())],
           "helped_mean_1to5": round(float(ans.drop_duplicates("participant").helped.mean()), 2),
           "creators_share": round(float((ans.drop_duplicates("participant").creates == "Sim").mean()), 2),
           "by_sector": ans.groupby("sector").correct.mean().round(3).to_dict(),
           "comments": [c for c in ans.drop_duplicates("participant").comment if c and c != "-"],
           "reference_tan2014_humans": config.HUMAN_BASELINE}
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "human_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    ans.to_csv(res / "human_test_answers.csv", index=False, encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
