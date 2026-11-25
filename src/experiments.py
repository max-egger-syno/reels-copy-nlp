"""Experiments E1 to E5. Every comparison is within creator; models are evaluated on each creator's newest reels."""
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .features import FEATURE_COLUMNS, portuguese_stopwords


def tfidf_pipeline(min_df: int = 2) -> Pipeline:
    return Pipeline([
        # Stopwords are kept so bigrams are real word sequences; stopword-only terms are dropped when ranking.
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
                                  token_pattern=r"(?u)\b[^\W\d_]{2,}\b")),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def feature_pipeline() -> Pipeline:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.5))


# ---------- E1: words that go viral vs words that do not, per creator ----------

def content_terms(vocab) -> np.ndarray:
    stop = portuguese_stopwords()
    return np.array([t for t in vocab if any(w not in stop for w in t.split())])


def word_ranking(texts: pd.Series, y: pd.Series, n_boot: int = 200, top_k: int = 15, keep: float = 0.70,
                 seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    texts, y = texts.reset_index(drop=True), y.astype(int).reset_index(drop=True)
    pos_idx, neg_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    hits = {"viral": {}, "flop": {}}
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos_idx, len(pos_idx)), rng.choice(neg_idx, len(neg_idx))]
        pipe = tfidf_pipeline()
        try:
            pipe.fit(texts.iloc[idx], y.iloc[idx])
        except ValueError:
            continue
        vocab = content_terms(pipe.named_steps["tfidf"].get_feature_names_out())
        proba = pipe.predict_proba(vocab)[:, 1]
        order = np.argsort(proba)
        for w in vocab[order[-top_k:]]:
            hits["viral"][w] = hits["viral"].get(w, 0) + 1
        for w in vocab[order[:top_k]]:
            hits["flop"][w] = hits["flop"].get(w, 0) + 1
    full = tfidf_pipeline().fit(texts, y)
    vocab = content_terms(full.named_steps["tfidf"].get_feature_names_out())
    p_full = dict(zip(vocab, full.predict_proba(vocab)[:, 1]))
    rows = []
    for direction, counts in hits.items():
        for w, c in counts.items():
            if c / n_boot >= keep:
                docs = texts.str.contains(rf"\b{w}\b", regex=True)
                rows.append({"term": w, "direction": direction, "stability": c / n_boot,
                             "p_viral": p_full.get(w, np.nan),
                             "n_docs_viral": int((docs & (y == 1)).sum()),
                             "n_docs_flop": int((docs & (y == 0)).sum())})
    return pd.DataFrame(rows)


def e1_words(df: pd.DataFrame, label: str = "y_viral", n_boot: int = 200) -> pd.DataFrame:
    out = []
    for user, g in df[df[label].notna()].groupby("username"):
        if g[label].nunique() < 2:
            continue
        r = word_ranking(g["text_all"], g[label], n_boot=n_boot)
        if not r.empty:
            r.insert(0, "username", user)
            r.insert(1, "sector", g["sector"].iloc[0])
            r.insert(2, "label", label)
            out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------- E2: copy devices, emotion and catchphrase vs lift, within creator ----------

def e2_correlations(df: pd.DataFrame, outcome: str = "lift") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (user, sector), g in df.groupby(["username", "sector"]):
        for f in FEATURE_COLUMNS:
            if f not in g:
                continue
            sub = g[[f, outcome]].dropna()
            if len(sub) < 15 or sub[f].nunique() < 2:
                continue
            rho, p = spearmanr(sub[f], sub[outcome])
            rows.append({"username": user, "sector": sector, "feature": f, "outcome": outcome,
                         "rho": rho, "p": p, "n": len(sub), "prevalence": sub[f].mean()})
    per_creator = pd.DataFrame(rows)
    summary = []
    for f, g in per_creator.groupby("feature"):
        n_pos, n_neg = int((g["rho"] > 0).sum()), int((g["rho"] < 0).sum())
        p_sign = binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if n_pos + n_neg else np.nan
        summary.append({"feature": f, "outcome": outcome, "n_creators": len(g), "n_positive": n_pos,
                        "n_negative": n_neg, "mean_rho": g["rho"].mean(), "sign_test_p": p_sign})
    return per_creator, pd.DataFrame(summary).sort_values("sign_test_p")


# ---------- E3: which of two reels by the same creator did better? ----------

def pair_correct(scores: pd.Series, pairs: pd.DataFrame, rng) -> np.ndarray:
    sa, sb = scores.reindex(pairs["reel_a"]).to_numpy(), scores.reindex(pairs["reel_b"]).to_numpy()
    pred = np.where(sa == sb, rng.random(len(sa)) < 0.5, sa > sb).astype(int)
    return (pred == pairs["label"].to_numpy()).astype(int)


def summarize(name: str, correct: np.ndarray, pairs: pd.DataFrame, n_boot: int = 1000, seed: int = 42,
              human: float = 0.613, tan: float = 0.656) -> dict:
    rng = np.random.default_rng(seed)
    users = pairs["username"].to_numpy()
    uniq = np.unique(users)
    by_user = {u: correct[users == u] for u in uniq}
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(uniq, len(uniq))
        c = np.concatenate([by_user[u] for u in sample])
        boots.append(c.mean())
    k, n = int(round(correct.sum())), len(correct)
    return {"model": name, "accuracy": k / n, "n_pairs": n, "n_creators": len(uniq),
            "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
            "p_vs_chance": binomtest(k, n, 0.5, alternative="greater").pvalue,
            "above_human_0.613": k / n > human, "above_tan_0.656": k / n > tan}


def _reel_scores(model, X_train, y_train, X_all, index) -> pd.Series:
    model.fit(X_train, y_train)
    return pd.Series(model.predict_proba(X_all)[:, 1], index=index)


def _per_creator_scores(df: pd.DataFrame, train: pd.DataFrame, label: str) -> pd.Series:
    """One TF-IDF + LR per creator, trained only on that creator's older reels (the professor's within-creator
    framing). Creators with a single class in training get a constant score, so their pairs fall back to a coin."""
    out = pd.Series(0.5, index=df.index)
    for user, tr in train.groupby("username"):
        if tr[label].nunique() < 2:
            continue
        mine = df[df["username"] == user]
        out.loc[mine.index] = _reel_scores(tfidf_pipeline(min_df=1), tr["text_all"], tr[label].astype(int),
                                           mine["text_all"], mine.index)
    return out


def e3_prediction(df: pd.DataFrame, pairs: pd.DataFrame, embeddings: np.ndarray | None = None,
                  label: str = "y_viral", seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    df = df.set_index("reel_id", drop=False)
    train = df[(df["split"] == "train") & df[label].notna()]
    test_pairs = pairs[pairs["split"] == "test"].reset_index(drop=True)
    y = train[label].astype(int)
    scores = {
        "random": pd.Series(rng.random(len(df)), index=df.index),
        "length_only": _reel_scores(feature_pipeline(), train[["n_caption_words", "n_opening_words"]], y,
                                    df[["n_caption_words", "n_opening_words"]], df.index),
        "devices_anew_perplexity_lr": _reel_scores(feature_pipeline(), train[FEATURE_COLUMNS], y,
                                                   df[FEATURE_COLUMNS], df.index),
        "tfidf_lr": _reel_scores(tfidf_pipeline(), train["text_all"], y, df["text_all"], df.index),
        "tfidf_lr_opening_only": _reel_scores(tfidf_pipeline(), train["opening"], y, df["opening"], df.index),
        "tfidf_lr_per_creator": _per_creator_scores(df, train, label),
    }
    if embeddings is not None:
        emb = pd.DataFrame(embeddings, index=df.index)
        scores["bertimbau_frozen_lr"] = _reel_scores(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05)),
            emb.loc[train.index], y, emb, df.index)

    rows, per_sector = [], []
    for name, s in scores.items():
        if name == "random":  # one draw on ~150 pairs can land anywhere in 0.42-0.58; report the expected level
            correct = np.mean([pair_correct(pd.Series(rng.random(len(df)), index=df.index), test_pairs, rng)
                               for _ in range(500)], axis=0)
        else:
            correct = pair_correct(s, test_pairs, rng)
        rows.append(summarize(name, correct, test_pairs))
        for sector, idx in test_pairs.groupby("sector").groups.items():
            c = correct[np.asarray(list(idx))]
            per_sector.append({"model": name, "sector": sector, "accuracy": c.mean(), "n_pairs": len(c)})
        test = df[(df["split"] == "test") & df[label].notna()]
        rows[-1]["reel_balanced_accuracy"] = balanced_accuracy_score(test[label].astype(int),
                                                                     (s.loc[test.index] > 0.5).astype(int))
    return pd.DataFrame(rows), pd.DataFrame(per_sector), scores


def learning_curve(df: pd.DataFrame, pairs: pd.DataFrame, embeddings: np.ndarray | None = None,
                   label: str = "y_viral", fractions=(0.2, 0.4, 0.6, 0.8, 1.0), repeats: int = 5,
                   seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.set_index("reel_id", drop=False)
    train = df[(df["split"] == "train") & df[label].notna()]
    test_pairs = pairs[pairs["split"] == "test"].reset_index(drop=True)
    emb = pd.DataFrame(embeddings, index=df.index) if embeddings is not None else None
    rows = []
    for frac, rep in product(fractions, range(repeats)):
        sub = train.sample(frac=frac, random_state=int(rng.integers(1e9)))
        if sub[label].nunique() < 2:
            continue
        y = sub[label].astype(int)
        models = {"tfidf_lr": (tfidf_pipeline(), sub["text_all"], df["text_all"])}
        if emb is not None:
            models["bertimbau_frozen_lr"] = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05)),
                                             emb.loc[sub.index], emb)
        for name, (model, Xtr, Xall) in models.items():
            model.fit(Xtr, y)
            s = pd.Series(model.predict_proba(Xall)[:, 1], index=df.index)
            rows.append({"model": name, "fraction": frac, "n_train_reels": len(sub), "repeat": rep,
                         "train_accuracy": model.score(Xtr, y),
                         "test_pair_accuracy": pair_correct(s, test_pairs, rng).mean()})
    return pd.DataFrame(rows)


# ---------- E4: do one creator's words work for another creator? ----------

def e4_transfer(df: pd.DataFrame, e1: pd.DataFrame | None = None, label: str = "y_viral",
                n_boot: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    lab = df[df[label].notna()]
    sector = lab.drop_duplicates("username").set_index("username")["sector"]
    users = sorted(lab["username"].unique())
    cells = []
    for a, b in product(users, users):
        tr = lab[(lab["username"] == a) & (lab["split"] == "train")]
        te = lab[(lab["username"] == b) & (lab["split"] == "test")]
        if tr[label].nunique() < 2 or te[label].nunique() < 2:
            continue
        pipe = tfidf_pipeline(min_df=1).fit(tr["text_all"], tr[label].astype(int))
        acc = balanced_accuracy_score(te[label].astype(int), pipe.predict(te["text_all"]))
        relation = "same_creator" if a == b else ("same_sector" if sector[a] == sector[b] else "other_sector")
        cells.append({"train_creator": a, "test_creator": b, "relation": relation, "balanced_accuracy": acc})
    matrix = pd.DataFrame(cells)
    summary = []
    for rel, g in matrix.groupby("relation"):
        v = g["balanced_accuracy"].to_numpy()
        boots = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
        summary.append({"relation": rel, "mean_balanced_accuracy": v.mean(), "n_cells": len(v),
                        "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5)})
    summary = pd.DataFrame(summary)
    if e1 is not None and not e1.empty:
        sets = {u: set(g["term"]) for u, g in e1[e1["direction"] == "viral"].groupby("username")}
        jac = []
        for a, b in product(sets, sets):
            if a < b:
                inter, union = len(sets[a] & sets[b]), len(sets[a] | sets[b]) or 1
                jac.append({"relation": "same_sector" if sector.get(a) == sector.get(b) else "other_sector",
                            "jaccard": inter / union})
        if jac:
            j = pd.DataFrame(jac).groupby("relation")["jaccard"].mean()
            summary["viral_word_jaccard"] = summary["relation"].map(j)
    return matrix, summary


# ---------- E5: topic (what it is about) vs wording (how it is said), per sector ----------

def e5_topics(df: pd.DataFrame, pairs: pd.DataFrame, k_range=range(2, 9), label: str = "y_viral",
              seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    choice, topics, compare = [], [], []
    for sector, g in df.groupby("sector"):
        g = g.set_index("reel_id", drop=False)
        tr, te = g[g["split"] == "train"], g[g["split"] == "test"]
        vect = CountVectorizer(stop_words=sorted(portuguese_stopwords()), min_df=2, max_df=0.5, max_features=1000,
                               token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
        Xtr = vect.fit_transform(tr["text_all"])
        Xte, Xall = vect.transform(te["text_all"]), vect.transform(g["text_all"])
        best = None
        for k in k_range:
            lda = LatentDirichletAllocation(n_components=k, random_state=seed).fit(Xtr)
            perp = lda.perplexity(Xte)
            choice.append({"sector": sector, "n_topics": k, "test_perplexity": perp})
            if best is None or perp < best[0]:
                best = (perp, k, lda)
        _, k, lda = best
        theta = pd.DataFrame(lda.transform(Xall), index=g.index, columns=[f"topic_{i}" for i in range(k)])
        vocab = vect.get_feature_names_out()
        for t in range(k):
            top = vocab[np.argsort(lda.components_[t])[::-1][:10]]
            rhos = []
            for user, gu in g.groupby("username"):
                rho, _ = spearmanr(theta.loc[gu.index, f"topic_{t}"], gu["lift"])
                rhos.append(rho)
            topics.append({"sector": sector, "n_topics": k, "topic": t, "top_words": ", ".join(top),
                           "mean_within_creator_rho_lift": np.nanmean(rhos)})
        lab_tr = tr[tr[label].notna()]
        sp = pairs[(pairs["split"] == "test") & (pairs["sector"] == sector)].reset_index(drop=True)
        if lab_tr[label].nunique() < 2 or sp.empty:
            continue
        y = lab_tr[label].astype(int)
        s_topic = _reel_scores(feature_pipeline(), theta.loc[lab_tr.index], y, theta, g.index)
        s_tfidf = _reel_scores(tfidf_pipeline(), lab_tr["text_all"], y, g["text_all"], g.index)
        compare.append({"sector": sector, "n_topics": k, "n_test_pairs": len(sp),
                        "topic_only_pair_accuracy": pair_correct(s_topic, sp, rng).mean(),
                        "tfidf_pair_accuracy": pair_correct(s_tfidf, sp, rng).mean()})
    return pd.DataFrame(choice), pd.DataFrame(topics), pd.DataFrame(compare)
