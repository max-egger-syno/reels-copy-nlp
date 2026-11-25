"""Run the full pipeline: ingest, (audio, transcription), outcomes, features, experiments E1 to E5, figures.

Usage:
  python run_all.py                       # real data in data/raw, creators in data/creators.csv
  python run_all.py --audio --transcribe  # also download audio and run Whisper
  python run_all.py --root tests/synthetic --no-bert
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

import config as C
from src import experiments as E
from src import figures as F
from src.features import build_features
from src.ingest import build_tables
from src.outcomes import build_pairs, compute_outcomes


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.ROOT), help="folder with data/raw and data/creators.csv")
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--no-bert", action="store_true")
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_N)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    data, res, figs = root / "data", root / "results", root / "figures"
    for d in (data / "processed", res, figs):
        d.mkdir(parents=True, exist_ok=True)
    proc = data / "processed"

    log("ingest")
    reels, creators = build_tables(data / "raw", data / "creators.csv", proc)
    log(f"{len(reels)} reels from {len(creators)} creators")

    transcripts_file = proc / "transcripts.parquet"
    if args.audio:
        from src.audio import download_audio
        log("audio download")
        audio = download_audio(reels, C.AUDIO_DIR)
        audio.to_parquet(proc / "audio_status.parquet", index=False)
        log(audio["audio_status"].str.split(":").str[0].value_counts().to_dict())
    if args.transcribe:
        from src.transcribe import transcribe_all
        log("transcription")
        audio = pd.read_parquet(proc / "audio_status.parquet")
        transcribe_all(audio, transcripts_file, C.WHISPER_MODEL, C.HOOK_SECONDS)
    if transcripts_file.exists():
        tr = pd.read_parquet(transcripts_file).drop_duplicates("reel_id", keep="last")
        reels = reels.merge(tr, on="reel_id", how="left")

    log("outcomes")
    df = compute_outcomes(reels, C.MIN_AGE_DAYS, C.BASELINE_WINDOW, C.LABEL_QUANTILE, C.TRAIN_FRACTION,
                          C.MIN_REELS_PER_CREATOR)
    pairs = build_pairs(df, C.PAIR_MAX_DAYS, C.PAIR_MIN_RATIO, C.SEED)
    log(f"{len(df)} usable reels, {df.username.nunique()} creators, pairs: {pairs.groupby('split').size().to_dict()}")

    log("features")
    df = build_features(df, C.LEXICON_DIR / "anew_br.csv")
    df.to_parquet(proc / "dataset_v1.parquet", index=False)
    pairs.to_parquet(proc / "pairs_v1.parquet", index=False)

    emb = None
    if not args.no_bert:
        try:
            from src.bert_embed import embed_texts
            log("BERTimbau embeddings")
            emb = embed_texts(df["text_all"], proc / "bert_cls.parquet")
        except Exception as e:
            log(f"BERT skipped: {type(e).__name__}: {e}")

    log("E1 words")
    e1 = pd.concat([E.e1_words(df, "y_viral", args.n_boot), E.e1_words(df, "y_convert", args.n_boot)],
                   ignore_index=True)
    e1.to_csv(res / "E1_words.csv", index=False)

    log("E2 correlations")
    pc_l, sum_l = E.e2_correlations(df, "lift")
    pc_c, sum_c = E.e2_correlations(df, "comments_per_view")
    pd.concat([pc_l, pc_c]).to_csv(res / "E2_per_creator.csv", index=False)
    pd.concat([sum_l, sum_c]).to_csv(res / "E2_summary.csv", index=False)

    log("E3 prediction")
    e3, e3_sector, _ = E.e3_prediction(df, pairs, emb)
    e3.to_csv(res / "E3_pairwise.csv", index=False)
    e3_sector.to_csv(res / "E3_by_sector.csv", index=False)
    lc = E.learning_curve(df, pairs, emb)
    lc.to_csv(res / "E3_learning_curve.csv", index=False)

    log("E4 transfer")
    matrix, e4 = E.e4_transfer(df, e1[e1["label"] == "y_viral"] if not e1.empty else None)
    matrix.to_csv(res / "E4_matrix.csv", index=False)
    e4.to_csv(res / "E4_summary.csv", index=False)

    log("E5 topics")
    k_choice, topics, compare = E.e5_topics(df, pairs)
    k_choice.to_csv(res / "E5_topic_perplexity.csv", index=False)
    topics.to_csv(res / "E5_topics.csv", index=False)
    compare.to_csv(res / "E5_topic_vs_words.csv", index=False)

    log("figures")
    F.fig_e3(e3, figs / "fig_e3_pairwise_accuracy.png")
    F.fig_learning_curve(lc, figs / "fig_e3_learning_curve.png")
    F.fig_e4(matrix, creators, figs / "fig_e4_transfer_matrix.png")
    F.fig_e1(e1, figs / "fig_e1_words_by_sector.png")

    summary = {
        "n_reels": int(len(df)), "n_creators": int(df.username.nunique()),
        "pairs": pairs.groupby("split").size().to_dict(),
        "anew_coverage_mean": float(df["all_anew_coverage"].mean()),
        "E3": e3[["model", "accuracy", "ci_low", "ci_high", "p_vs_chance"]].round(3).to_dict("records"),
        "E4": e4.round(3).to_dict("records"),
    }
    (res / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log("done")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
