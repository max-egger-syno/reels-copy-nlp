"""Extract the ANEW-Br norms (Kristensen et al., 2011, Annex 1) from the article PDF."""
import re
import sys
from pathlib import Path

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LEXICON_DIR

NUM = re.compile(r"^\d+,\d+$")


def extract(pdf_path: str) -> pd.DataFrame:
    doc = pymupdf.open(pdf_path)
    tokens = []
    started = False
    for page in doc:
        text = page.get_text("text")
        if not started:
            idx = text.find("Anexo 1")
            if idx < 0:
                continue
            started = True
            text = text[idx:]
        tokens.extend(t.strip() for t in re.split(r"[\t\n]+", text) if t.strip())

    rows = []
    i = 0
    while i < len(tokens) - 4:
        word = tokens[i]
        nums = tokens[i + 1:i + 5]
        if (re.fullmatch(r"[A-ZÀ-ÜÇ][A-ZÀ-ÜÇ\- ]*", word) and all(NUM.match(n) for n in nums)):
            vals = [float(n.replace(",", ".")) for n in nums]
            rows.append({"word": word.lower(), "valence_mean": vals[0], "valence_sd": vals[1],
                         "arousal_mean": vals[2], "arousal_sd": vals[3]})
            i += 5
        else:
            i += 1
    df = pd.DataFrame(rows)
    print(f"entries in annex: {len(df)}")
    # The annex lists GATO twice; duplicates are averaged.
    return df.groupby("word", as_index=False).mean()


if __name__ == "__main__":
    pdf = sys.argv[1]
    df = extract(pdf)
    out = LEXICON_DIR / "anew_br.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"rows={len(df)}")
    print(f"valence {df.valence_mean.min()}..{df.valence_mean.max()}  arousal {df.arousal_mean.min()}..{df.arousal_mean.max()}")
    print(df.sample(8, random_state=1).to_string(index=False))
