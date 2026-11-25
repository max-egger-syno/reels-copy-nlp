"""Turn results/ into the deliverables that depend on numbers:
  report/numbers.tex   LaTeX macros the report reads (so the PDF always matches the latest run)
  cards/copy_cards.md  per-sector copy cards (English, Portuguese examples)
  RESUMO.md            short Portuguese summary for the team
Run after run_all.py (and after scripts/score_human_test.py when responses exist)."""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

RES, PROC = ROOT / "results", config.PROCESSED_DIR
SECTOR_EN = {"ai": "AI", "hair_visagism": "Hair \\& visagism", "sales": "Sales", "trading": "Trading", "sports": "Sports"}
SECTOR_MD = {k: v.replace("\\&", "&") for k, v in SECTOR_EN.items()}
SECTOR_PT = {"ai": "IA", "hair_visagism": "Cabelo e visagismo", "sales": "Vendas", "trading": "Trading", "sports": "Esporte"}
FEATURE_EN = {
    "dev_question": "a question", "dev_number_list": "a numbered list", "dev_money_number": "money figures",
    "dev_prohibition": "a prohibition (\"never do\")", "dev_curiosity_gap": "a curiosity gap", "dev_contrast": "a contrast",
    "dev_second_person": "second person (\"você\")", "dev_first_person_story": "a first-person story",
    "dev_imperative": "an imperative", "dev_urgency": "urgency words", "dev_social_proof": "social proof",
    "dev_cta_comment": "a comment CTA", "dev_cta_save": "a save CTA", "dev_cta_share": "a share CTA",
    "dev_cta_link": "a link CTA", "dev_cta_follow": "a follow CTA", "opening_question": "an opening question",
    "open_valence": "opening valence (ANEW-Br)", "open_arousal": "opening arousal (ANEW-Br)",
    "all_valence": "overall valence (ANEW-Br)", "all_arousal": "overall arousal (ANEW-Br)",
    "n_caption_words": "caption length", "n_opening_words": "opening length", "n_hashtags": "hashtag count",
    "n_emojis": "emoji count", "log_opening_perplexity": "opening perplexity (catchphrase-ness)"}


def read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pval(p):
    return "--" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def tex_escape(v):
    s = str(v).replace("%", "\\%").replace("_", " ").replace("&", "\\&")
    return re.sub(r'"([^"]*)"', r"``\1''", s)


def macro_name(s):
    return re.sub(r"[^A-Za-z]", "", s.title())


def main():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(PROC / "dataset_v1.parquet")
    e1, e2s, e2p = read("E1_words.csv"), read("E2_summary.csv"), read("E2_per_creator.csv")
    e3, e3s, lc = read("E3_pairwise.csv"), read("E3_by_sector.csv"), read("E3_learning_curve.csv")
    e4, e5 = read("E4_summary.csv"), read("E5_topic_vs_words.csv")
    human = json.loads((RES / "human_test.json").read_text(encoding="utf-8")) if (RES / "human_test.json").exists() else None

    m = {}  # macro name -> value
    m["nReels"] = summary["n_reels"]
    m["nCreators"] = summary["n_creators"]
    m["nTestPairs"] = summary["pairs"].get("test", 0)
    m["nTrainPairs"] = summary["pairs"].get("train", 0)
    m["nTranscribed"] = int((df["n_spoken_words"].fillna(0) > 0).sum()) if "n_spoken_words" in df else 0
    m["medianViews"] = f"{int(df['views'].median()):,}"
    posted = pd.to_datetime(df["posted_at"], utc=True)
    m["dateFrom"], m["dateTo"] = posted.min().strftime("%b %Y"), posted.max().strftime("%b %Y")
    m["anewCoverage"] = fmt(summary.get("anew_coverage_mean", np.nan) * 100, 0)
    for r in e3.itertuples():
        k = macro_name(r.model)
        m[f"acc{k}"], m[f"lo{k}"], m[f"hi{k}"], m[f"p{k}"] = fmt(r.accuracy), fmt(r.ci_low), fmt(r.ci_high), pval(r.p_vs_chance)
    for r in e4.itertuples():
        m[f"transfer{macro_name(r.relation)}"] = fmt(r.mean_balanced_accuracy)
    best = e3[e3.model != "random"].sort_values("accuracy", ascending=False).iloc[0]
    m["bestModel"] = best.model.replace("_", " ")
    m["accBest"] = fmt(best.accuracy)
    if not e5.empty:
        m["topicAcc"], m["tfidfSectorAcc"] = fmt(e5.topic_only_pair_accuracy.mean()), fmt(e5.tfidf_pair_accuracy.mean())
    lift = e2s[e2s.outcome == "lift"].sort_values("sign_test_p") if not e2s.empty else e2s
    for i, r in enumerate(lift.head(3).itertuples()):
        tag = "ABC"[i]
        m[f"devName{tag}"] = FEATURE_EN.get(r.feature, r.feature)
        m[f"devSign{tag}"] = f"{r.n_negative if r.mean_rho < 0 else r.n_positive}/{r.n_creators}"
        m[f"devDir{tag}"] = "lower" if r.mean_rho < 0 else "higher"
        m[f"devP{tag}"] = pval(r.sign_test_p)
    if not lc.empty:
        g = lc.groupby(["model", "fraction"]).test_pair_accuracy.mean().unstack(0)
        m["lcSmall"], m["lcFull"] = fmt(g.iloc[0].max()), fmt(g.iloc[-1].max())
    if human:
        m["humanN"] = human["n_participants"]
        m["humanBefore"], m["humanAfter"] = fmt(human["accuracy_before_cards"]), fmt(human["accuracy_after_cards"])
        m["humanAll"], m["humanGain"] = fmt(human["accuracy_all"]), fmt(human["mean_gain_pp"], 1)
        m["humanGainLo"], m["humanGainHi"] = (fmt(v, 1) for v in human["gain_ci95_pp"])
        m["humanHelped"] = fmt(human["helped_mean_1to5"], 1)
    else:
        for k in ("humanN", "humanBefore", "humanAfter", "humanAll", "humanGain", "humanGainLo", "humanGainHi", "humanHelped"):
            m[k] = "--"
    (ROOT / "report").mkdir(exist_ok=True)
    tex = "% generated by scripts/make_outputs.py, do not edit by hand\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{tex_escape(v)}}}\n" for k, v in m.items())
    (ROOT / "report" / "numbers.tex").write_text(tex, encoding="utf-8")
    # The report folder must be self-contained so it can be uploaded to Overleaf as is.
    for fig in (ROOT / "figures").glob("*.png"):
        shutil.copy2(fig, ROOT / "report" / fig.name)

    # ---------- copy cards ----------
    cards = ["# Copy cards: what goes viral within each creator\n",
             "Built from within-creator comparisons: each creator's most viral reels (top 30% of lift over their own "
             "rolling median) against their least viral (bottom 30%). Words are kept only if they rank in the top 15 "
             f"in at least 70% of {config.BOOTSTRAP_N} bootstrap refits. Examples are real Portuguese copy from the "
             "creator's own viral reels.\n"]
    viral = e1[(e1.label == "y_viral")] if not e1.empty else e1
    for sector in config.SECTORS:
        sub = df[df.sector == sector]
        if sub.empty:
            continue
        cards.append(f"\n## {SECTOR_MD[sector]}  ({', '.join('@' + u for u in sorted(sub.username.unique()))})\n")
        acc = e3s[(e3s.model == "tfidf_lr_per_creator") & (e3s.sector == sector)] if not e3s.empty else e3s
        if not acc.empty:
            cards.append(f"Per-creator model picks the more viral reel in {acc.accuracy.iloc[0]:.0%} of "
                         f"{int(acc.n_pairs.iloc[0])} held-out pairs.\n")
        w = viral[viral.sector == sector] if not viral.empty else viral
        up = w[w.direction == "viral"].sort_values("stability", ascending=False).head(6)
        down = w[w.direction == "flop"].sort_values("stability", ascending=False).head(6)
        if not up.empty:
            cards.append("**Words that ride with virality:** " + ", ".join(f"`{t}` (@{u})" for t, u in zip(up.term, up.username)) + "\n")
        if not down.empty:
            cards.append("**Words that ride with flops:** " + ", ".join(f"`{t}` (@{u})" for t, u in zip(down.term, down.username)) + "\n")
        sec = e2p[(e2p.sector == sector) & (e2p.outcome == "lift")] if not e2p.empty else e2p
        if not sec.empty:
            agg = sec.groupby("feature").rho.agg(["mean", "count"])
            agg = agg[agg["count"] == sec.username.nunique()].sort_values("mean")
            if len(agg):
                lo, hi = agg.index[0], agg.index[-1]
                cards.append(f"**Devices that agree across this sector's creators:** higher lift with {FEATURE_EN.get(hi, hi)} "
                             f"(mean rho {agg.loc[hi, 'mean']:+.2f}); lower lift with {FEATURE_EN.get(lo, lo)} "
                             f"(mean rho {agg.loc[lo, 'mean']:+.2f}).\n")
        top = sub[sub.y_viral == 1].sort_values("lift", ascending=False)
        ok = lambda s: len(s.split()) >= 6 and sum(c.isdigit() for c in s) <= 2 and "@" not in s
        ex = top["opening"].dropna().map(lambda s: s.strip()).loc[lambda s: s.map(ok)].head(2)
        for e in ex:
            cards.append(f"> \"{e[:160]}\"\n")
    (ROOT / "cards").mkdir(exist_ok=True)
    (ROOT / "cards" / "copy_cards.md").write_text("\n".join(cards), encoding="utf-8")

    # ---------- Portuguese summary ----------
    lines = ["# Resumo dos resultados\n", f"Gerado automaticamente a partir de `results/` ({pd.Timestamp.now():%d/%m %H:%M}).\n",
             "## Dados",
             f"- {m['nReels']} Reels utilizáveis de {m['nCreators']} criadores ({m['dateFrom']} a {m['dateTo']}), "
             f"{m['nTranscribed']} com fala transcrita. Mediana de {m['medianViews']} views.",
             f"- Pares de teste (mesmo criador, até 14 dias, views 1,5x ou mais): {m['nTestPairs']}. Pares de treino: {m['nTrainPairs']}.\n",
             "## E3: o modelo acerta qual Reel do par viralizou mais?",
             "| Modelo | Acerto | IC 95% | p contra 50% |", "|---|---|---|---|"]
    for r in e3.sort_values("accuracy", ascending=False).itertuples():
        lines.append(f"| {r.model} | {r.accuracy:.1%} | {r.ci_low:.2f} a {r.ci_high:.2f} | {pval(r.p_vs_chance)} |")
    lines.append(f"\nReferência (Tan et al. 2014, tweets): humanos 61,3%, modelo 65,6%.\n")
    if not e3s.empty:
        lines.append("Por setor (modelo por criador): " + ", ".join(
            f"{SECTOR_PT[r.sector]} {r.accuracy:.0%} ({r.n_pairs} pares)" for r in e3s[e3s.model == "tfidf_lr_per_creator"].itertuples()) + "\n")
    if not lift.empty:
        lines.append("## E2: recursos de copy contra o lift, dentro de cada criador")
        for r in lift.head(5).itertuples():
            lines.append(f"- {r.feature}: positivo em {r.n_positive}, negativo em {r.n_negative} de {r.n_creators} criadores "
                         f"(rho médio {r.mean_rho:+.3f}, teste do sinal p={pval(r.sign_test_p)})")
        lines.append("")
    if not e4.empty:
        lines.append("## E4: as palavras de um criador servem para outro?")
        for r in e4.itertuples():
            lines.append(f"- {r.relation}: acurácia balanceada {r.mean_balanced_accuracy:.2f} ({r.n_cells} combinações)")
        lines.append("")
    if not e5.empty:
        lines.append(f"## E5: tema contra palavras\n- Só tema (LDA): {m['topicAcc']}; palavras (TF-IDF): {m['tfidfSectorAcc']} (média dos setores)\n")
    lines.append("## Teste com pessoas")
    lines.append(f"- {m['humanN']} participantes; antes dos cartões {m['humanBefore']}, depois {m['humanAfter']} "
                 f"(ganho {m['humanGain']} p.p., IC {m['humanGainLo']} a {m['humanGainHi']}); nota média das dicas {m['humanHelped']}/5."
                 if human else "- Ainda sem respostas: cole as mensagens em `data/human_test_responses.txt` e rode `python scripts/score_human_test.py`.")
    lines.append("\n## Arquivos\n- Relatório: `report/main.tex` (números em `report/numbers.tex`), figuras em `figures/`.\n- Cartões: `cards/copy_cards.md`.")
    (ROOT / "RESUMO.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote report/numbers.tex, cards/copy_cards.md, RESUMO.md")


if __name__ == "__main__":
    main()
