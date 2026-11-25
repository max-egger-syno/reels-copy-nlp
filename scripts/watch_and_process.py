"""Watch the Downloads folder for collected Reels files, copy them into data/raw, download audio right away
(video URLs expire) and transcribe with parallel faster-whisper workers. Resumable; safe to restart."""
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.audio import download_audio  # noqa: E402
from src.ingest import load_raw  # noqa: E402

DOWNLOADS = Path.home() / "Downloads"
TRANSCRIPTS = config.PROCESSED_DIR / "transcripts.parquet"
LOG = config.MEDIA_DIR / "watch.log"
WORKERS, THREADS = 4, 4
MAX_HOURS = 6
RETRY_AFTER_S, MAX_TRIES = 600, 4  # audio failures are often a temporary CDN block

_model = None


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def file_quality(path: Path):
    try:
        p = json.loads(path.read_text(encoding="utf-8"))
        details = sum(1 for i in p.get("items", []) if i.get("__details"))
        return details, p.get("collected_at", ""), p.get("username", "").lower()
    except Exception:
        return -1, "", ""


def _payload_quality(p: dict):
    details = sum(1 for i in p.get("items", []) if i.get("__details"))
    return details, p.get("collected_at", ""), p.get("username", "").lower()


def candidate_payloads():
    """Yield payloads from per-profile files and from 'Export all' bundles (all_reels_*.json)."""
    for f in DOWNLOADS.glob("*reels_*.json"):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for run in (p.get("runs") or [p]):
            if isinstance(run, dict) and run.get("items"):
                yield run


def sync_downloads(creators):
    best = {}
    for run in candidate_payloads():
        details, collected, user = _payload_quality(run)
        if user in creators and (user not in best or (details, collected) > best[user][:2]):
            best[user] = (details, collected, run)
    changed = []
    for user, (details, collected, run) in best.items():
        dest = config.RAW_DIR / f"{user}_reels.json"
        legacy = list(config.RAW_DIR.glob(f"{user}_reels_*.json"))
        current = max([file_quality(p)[:2] for p in [dest, *legacy] if p.exists()], default=(-1, ""))
        if (details, collected) > current:
            for p in legacy:
                p.unlink()
            dest.write_text(json.dumps(run), encoding="utf-8")
            changed.append(f"{user} ({details} with details)")
    return changed


def _init_worker():
    global _model
    from faster_whisper import WhisperModel
    _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=THREADS)


def _transcribe(args):
    reel_id, path = args
    from src.transcribe import read_wav16k
    try:
        segments, info = _model.transcribe(read_wav16k(path), language="pt", word_timestamps=True, vad_filter=True,
                                           beam_size=1)
        words = [(w.word.strip(), w.start, w.end) for s in segments for w in (s.words or [])]
        return {"reel_id": reel_id, "transcript": " ".join(w for w, _, _ in words),
                "hook": " ".join(w for w, st, _ in words if st < config.HOOK_SECONDS),
                "n_spoken_words": len(words),
                "speech_seconds": (words[-1][2] - words[0][1]) if words else 0.0,
                "audio_seconds": info.duration, "error": ""}
    except Exception as e:
        return {"reel_id": reel_id, "transcript": "", "hook": "", "n_spoken_words": 0, "speech_seconds": 0.0,
                "audio_seconds": 0.0, "error": f"{type(e).__name__}: {e}"[:200]}


def transcribe_pending(audio: pd.DataFrame, pool):
    done = pd.read_parquet(TRANSCRIPTS) if TRANSCRIPTS.exists() else pd.DataFrame(columns=["reel_id"])
    todo = audio[audio["audio_path"].notna() & ~audio["reel_id"].isin(done["reel_id"])]
    if todo.empty:
        return 0
    log(f"transcribing {len(todo)} reels with {WORKERS} workers")
    rows, n = [], 0
    for r in pool.imap_unordered(_transcribe, list(zip(todo.reel_id, todo.audio_path))):
        rows.append(r)
        n += 1
        if len(rows) >= 10:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(TRANSCRIPTS, index=False)
            rows = []
            log(f"  transcribed {n}/{len(todo)}")
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_parquet(TRANSCRIPTS, index=False)
    log(f"  transcribed {n}/{len(todo)} (batch done)")
    return n


def main():
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    creators = set(pd.read_csv(config.CREATORS_FILE)["username"].str.lower().str.lstrip("@"))
    log(f"watching {DOWNLOADS} for {len(creators)} creators")
    start = time.time()
    tries, last_try = {}, {}
    with mp.Pool(WORKERS, initializer=_init_worker) as pool:
        while time.time() - start < MAX_HOURS * 3600:
            changed = sync_downloads(creators)
            for c in changed:
                log(f"new file: {c}")
            reels = load_raw(config.RAW_DIR)
            reels = reels[reels["username"].isin(creators)]
            pending_audio = reels[~reels["reel_id"].map(lambda r: (config.AUDIO_DIR / f"{r}.wav").exists())
                                  & reels["reel_id"].map(lambda r: tries.get(r, 0) < MAX_TRIES
                                                         and time.time() - last_try.get(r, 0) > RETRY_AFTER_S)
                                  & reels["video_url"].notna()]
            if len(pending_audio):
                log(f"downloading audio for {len(pending_audio)} reels")
                st = download_audio(pending_audio, config.AUDIO_DIR)
                for rid in st.loc[st["audio_status"].str.startswith("error"), "reel_id"]:
                    tries[rid] = tries.get(rid, 0) + 1
                    last_try[rid] = time.time()
                log(f"  audio status: {st['audio_status'].value_counts().to_dict()}")
            audio = pd.DataFrame({"reel_id": reels["reel_id"]})
            audio["audio_path"] = audio["reel_id"].map(
                lambda r: str(config.AUDIO_DIR / f"{r}.wav") if (config.AUDIO_DIR / f"{r}.wav").exists() else None)
            transcribe_pending(audio, pool)
            have = set(reels["username"].unique())
            done = pd.read_parquet(TRANSCRIPTS) if TRANSCRIPTS.exists() else pd.DataFrame(columns=["reel_id"])
            left = audio["audio_path"].notna() & ~audio["reel_id"].isin(done["reel_id"])
            if creators <= have and not left.any():
                log(f"ALL DONE: {len(have)} creators, {len(reels)} reels, {len(done)} transcripts")
                return
            log(f"status: {len(have)}/{len(creators)} creators collected, {len(done)} transcripts; waiting")
            time.sleep(45)
    log("stopped: max runtime reached")


if __name__ == "__main__":
    main()
