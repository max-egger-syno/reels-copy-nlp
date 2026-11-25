"""Transcribe Reel audio with faster-whisper on CPU. Resumable: rows already in the output are skipped."""
import os
from pathlib import Path

import wave

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_wav16k(path: str) -> np.ndarray:
    # audio.py writes 16 kHz mono PCM; decoding here avoids faster-whisper's PyAV dependency.
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def transcribe_all(audio: pd.DataFrame, out_file: Path, model_name: str = "small", hook_seconds: float = 5.0,
                   save_every: int = 25) -> pd.DataFrame:
    from faster_whisper import WhisperModel

    done = pd.read_parquet(out_file) if out_file.exists() else pd.DataFrame(columns=["reel_id"])
    todo = audio[audio["audio_path"].notna() & ~audio["reel_id"].isin(done["reel_id"])]
    if todo.empty:
        return done
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    rows = []
    for i, r in enumerate(tqdm(todo.itertuples(), total=len(todo), desc="whisper")):
        segments, info = model.transcribe(read_wav16k(r.audio_path), language="pt", word_timestamps=True,
                                          vad_filter=True)
        words = [(w.word.strip(), w.start, w.end) for s in segments for w in (s.words or [])]
        rows.append({
            "reel_id": r.reel_id,
            "transcript": " ".join(w for w, _, _ in words),
            "hook": " ".join(w for w, start, _ in words if start < hook_seconds),
            "n_spoken_words": len(words),
            "speech_seconds": (words[-1][2] - words[0][1]) if words else 0.0,
            "audio_seconds": info.duration,
        })
        if (i + 1) % save_every == 0:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(out_file, index=False)
            rows = []
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_parquet(out_file, index=False)
    return done
