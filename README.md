# Which Words Make a Reel Travel? A Within-Creator Look at Portuguese Reels

Insper, Natural Language Processing (2026s2). Final report: [report/Final_Report.pdf](report/Final_Report.pdf).

For nine Brazilian Instagram creators in five sectors, each Reel is compared only with the same creator's
neighbouring Reels, so audience size drops out. The project asks which Portuguese words and copy devices go with the
Reels that beat the creator's own baseline, and tests whether a model can pick the more viral of two Reels by the
same creator (the pairwise setup of Tan, Lee and Pang, 2014).

## Deliverables

| Path | What it is |
|---|---|
| `report/` | 2-page report in the IEEE conference template (`main.tex`, figure, compiled PDF) |
| `cards/copy_cards.md` | Educational material: copy cards per sector, with real Portuguese examples |
| `notebook/analyze_your_profile.ipynb` | Educational material: run the same within-creator analysis on your own profile |
| `human_test/index.html` | Human test: 16 pairs judged before and after reading four tips |
| `results/`, `figures/` | All experiment outputs behind the report |

## Code

| Path | Role |
|---|---|
| `collection/collect_reels_snippet.js` | Browser snippet that saves a creator's public Reels metadata (views, likes, comments, caption, date) |
| `src/ingest.py` | Raw JSON to tables |
| `src/audio.py`, `src/transcribe.py` | Audio download and Whisper transcription; the opening is the first 5 s of speech |
| `src/outcomes.py` | Lift (log views minus the median of the 10 previous and 10 next Reels), labels, 70/30 temporal split, pairs |
| `src/features.py` | Regex copy devices, Brazilian ANEW emotion scores, per-creator bigram perplexity |
| `src/bert_embed.py` | Frozen BERTimbau [CLS] embeddings |
| `src/experiments.py` | E1 word ranking, E2 device correlations, E3 pairwise prediction, E4 transfer, E5 LDA topics |
| `src/figures.py` | Figures |
| `run_all.py` | Full pipeline |
| `scripts/` | Audio and transcription runner, ANEW-Br extraction, human test build and scoring, report outputs |
| `tests/synthetic/make_synthetic.py` | Synthetic data with planted effects to check the pipeline |

## Run

```
pip install -r requirements.txt
python tests/synthetic/make_synthetic.py
python run_all.py --root tests/synthetic          # pipeline check on synthetic data
```

On real data: put the collected JSON files in `data/raw/`, list creators in `data/creators.csv` (`username,sector`),
build the lexicon once with `python scripts/extract_anew_br.py <ANEW-Br PDF>`, then run
`python run_all.py --audio --transcribe`. Audio is stored outside the project (`REELS_MEDIA_DIR`, default `~/reels-media`).

## Not included

Raw collected data, audio, transcripts and the ANEW-Br lexicon (third-party norms) are not published.
Human test answers are summarized in `results/human_test.json` without participant names.
