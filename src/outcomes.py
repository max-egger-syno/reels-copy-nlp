"""Within-creator performance measures, labels, temporal split and evaluation pairs."""
import numpy as np
import pandas as pd


def compute_outcomes(reels: pd.DataFrame, min_age_days: int = 14, window: int = 10, q: float = 0.30,
                     train_fraction: float = 0.70, min_reels: int = 60) -> pd.DataFrame:
    df = reels.copy()
    age = (pd.to_datetime(df["collected_at"], utc=True) - pd.to_datetime(df["posted_at"], utc=True)).dt.days
    df = df[(age >= min_age_days) & (df["views"] > 0)].copy()
    df = df.sort_values(["username", "posted_at"]).reset_index(drop=True)
    df["log_views"] = np.log(df["views"])

    parts = []
    for _, g in df.groupby("username", sort=False):
        if len(g) < min_reels:
            continue
        g = g.copy()
        g["t_index"] = np.arange(len(g))
        n_train = int(round(train_fraction * len(g)))
        g["split"] = np.where(g["t_index"] < n_train, "train", "test")
        # Baseline uses only neighbours from the same split, so test views never enter training targets.
        base = np.empty(len(g))
        for split in ("train", "test"):
            idx = np.flatnonzero(g["split"].to_numpy() == split)
            lv = g["log_views"].to_numpy()[idx]
            for k in range(len(idx)):
                neigh = np.r_[lv[max(0, k - window):k], lv[k + 1:k + 1 + window]]
                base[idx[k]] = np.median(neigh) if len(neigh) else lv[k]
        g["baseline"] = base
        g["lift"] = g["log_views"] - base
        g["comments_per_view"] = g["comments"] / g["views"]
        g["likes_per_view"] = g["likes"] / g["views"]
        train = g[g["split"] == "train"]
        for col, name in (("lift", "y_viral"), ("comments_per_view", "y_convert")):
            lo, hi = train[col].quantile(q), train[col].quantile(1 - q)
            g[name] = np.where(g[col] >= hi, 1.0, np.where(g[col] <= lo, 0.0, np.nan))
            g.loc[g[col].isna(), name] = np.nan
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def build_pairs(df: pd.DataFrame, max_days: int = 14, min_ratio: float = 1.5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (user, split), g in df.groupby(["username", "split"], sort=False):
        g = g.sort_values("posted_at")
        used = set()
        ts = pd.to_datetime(g["posted_at"], utc=True).to_numpy()
        ids, views = g["reel_id"].to_numpy(), g["views"].to_numpy()
        for i in range(len(g)):
            if ids[i] in used:
                continue
            best = None
            for j in range(i + 1, len(g)):
                gap = (ts[j] - ts[i]) / np.timedelta64(1, "D")
                if gap > max_days:
                    break
                if ids[j] in used:
                    continue
                ratio = max(views[i], views[j]) / max(min(views[i], views[j]), 1)
                if ratio >= min_ratio:
                    best = j
                    break
            if best is None:
                continue
            used.update((ids[i], ids[best]))
            a, b = (i, best) if rng.random() < 0.5 else (best, i)
            rows.append({"username": user, "split": split, "reel_a": ids[a], "reel_b": ids[b],
                         "label": int(views[a] > views[b])})
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        sectors = df.drop_duplicates("username").set_index("username")["sector"]
        pairs["sector"] = pairs["username"].map(sectors)
    return pairs
