import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LEXICON_DIR = DATA_DIR / "lexicons"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Audio stays outside the synced project folder.
MEDIA_DIR = Path(os.environ.get("REELS_MEDIA_DIR", Path.home() / "reels-media"))
AUDIO_DIR = MEDIA_DIR / "audio"

CREATORS_FILE = DATA_DIR / "creators.csv"  # username,sector

SECTORS = ["ai", "hair_visagism", "sales", "trading", "sports"]

WHISPER_MODEL = "small"
HOOK_SECONDS = 5.0
MIN_AGE_DAYS = 14
BASELINE_WINDOW = 10
LABEL_QUANTILE = 0.30
TRAIN_FRACTION = 0.70
PAIR_MAX_DAYS = 14
PAIR_MIN_RATIO = 1.5
MIN_REELS_PER_CREATOR = 50
BOOTSTRAP_N = 200
SEED = 42
HUMAN_BASELINE = 0.613
TAN_MODEL = 0.656

for d in (RAW_DIR, PROCESSED_DIR, LEXICON_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
