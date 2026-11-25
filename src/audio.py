"""Download Reel audio (video is discarded right away) into the media folder outside OneDrive."""
import os
import subprocess
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

HEADERS = {"User-Agent": "Mozilla/5.0"}


def download_audio(reels: pd.DataFrame, audio_dir: Path, timeout: int = 60) -> pd.DataFrame:
    audio_dir.mkdir(parents=True, exist_ok=True)
    status = []
    for _, r in tqdm(reels.iterrows(), total=len(reels), desc="audio"):
        wav = audio_dir / f"{r.reel_id}.wav"
        if wav.exists():
            status.append((r.reel_id, str(wav), "cached"))
            continue
        if not isinstance(r.video_url, str) or not r.video_url:
            status.append((r.reel_id, None, "no_url"))
            continue
        tmp = audio_dir / f"{r.reel_id}.{os.getpid()}.mp4"  # unique per process
        try:
            with requests.get(r.video_url, headers=HEADERS, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(1 << 16):
                        fh.write(chunk)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-vn", "-ac", "1",
                            "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
            status.append((r.reel_id, str(wav), "ok"))
        except Exception as e:  # expired URL, network error, no audio track
            code = getattr(getattr(e, "response", None), "status_code", "")
            status.append((r.reel_id, None, f"error: {type(e).__name__} {code}".strip()))
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return pd.DataFrame(status, columns=["reel_id", "audio_path", "audio_status"])
