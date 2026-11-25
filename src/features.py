"""Text cleaning and features: regex copy devices, ANEW-Br emotion, creator bigram perplexity."""
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@[\w.]+")
HASHTAG = re.compile(r"#[\wÀ-ſ]+")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
TOKEN = re.compile(r"[a-zà-öø-ÿ0-9]+(?:-[a-zà-öø-ÿ0-9]+)*", re.IGNORECASE)

DEVICES = {
    "question": r"\?|\b(você sabia|sabia que|já pensou|por que|porque será|como fazer|qual (é|o|a))\b",
    "number_list": r"\b\d+\s+(erros?|dicas?|passos?|formas?|maneiras?|motivos?|coisas?|segredos?|regras?|sinais?)\b|\b(top|primeiro|segundo|terceiro)\b",
    "money_number": r"r\$\s?\d|\b\d+\s?(mil|k|milh(ão|ões)|reais|%|por cento|dias|minutos)\b",
    "prohibition": r"\b(pare de|para de|nunca|não faça|não faz|evite|cuidado|erro|errado|ninguém deveria)\b",
    "curiosity_gap": r"\b(ninguém te conta|ninguém fala|segredo|o que ninguém|descobri|revelar|a verdade sobre|você não vai acreditar)\b",
    "contrast": r"\bnão é .{1,40}?, é\b|\b(na verdade|todo mundo (faz|acha|pensa))\b",
    "second_person": r"\b(você|vocês|seu|sua|seus|suas|te|tu)\b",
    "first_person_story": r"\b(eu (fiz|perdi|ganhei|descobri|comecei|aprendi|errei|testei)|quando eu|minha história)\b",
    "imperative": r"\b(faça|faz|olha|veja|assista|aprenda|descubra|aproveite|confira|presta atenção|escuta)\b",
    "urgency": r"\b(agora|hoje|últimas? vagas?|só até|antes que|corre|urgente|rápido)\b",
    "social_proof": r"\b(alunos?|clientes?|milhares de|todo mundo|comprovado|resultado(s)?|especialistas?)\b",
    "cta_comment": r"\b(comenta|comente|comentário|deixa (aqui )?nos comentários|escreve)\b",
    "cta_save": r"\b(salva|salve|guarda)\b",
    "cta_share": r"\b(compartilha|compartilhe|manda (pra|para)|envia (pra|para)|marca (alguém|um amigo|quem))\b",
    "cta_link": r"\b(link na bio|link da bio|clica no link|direct|dm)\b",
    "cta_follow": r"\b(segue|siga|me segue|sigam)\b",
}
DEVICE_RE = {k: re.compile(v, re.IGNORECASE) for k, v in DEVICES.items()}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def clean_text(s: str) -> str:
    s = s or ""
    s = URL.sub(" ", s)
    s = MENTION.sub(" ", s)
    s = HASHTAG.sub(" ", s)
    s = EMOJI.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def tokenize(s: str) -> list[str]:
    return TOKEN.findall((s or "").lower())


@lru_cache(maxsize=1)
def portuguese_stopwords() -> frozenset:
    try:
        from nltk.corpus import stopwords
        return frozenset(stopwords.words("portuguese"))
    except LookupError:
        return frozenset("a o as os de da do das dos e é em um uma para com que não no na nos nas por se mais mas "
                         "como ao aos à às ou eu você ele ela isso esse essa este esta já também só muito sua seu".split())


class Anew:
    def __init__(self, csv_path: Path):
        lex = pd.read_csv(csv_path)
        lex["key"] = lex["word"].map(strip_accents)
        self.exact = lex.groupby("key")[["valence_mean", "arousal_mean"]].mean().to_dict("index")
        from nltk.stem import RSLPStemmer
        self.stemmer = RSLPStemmer()
        lex["stem"] = lex["word"].map(lambda w: strip_accents(self.stemmer.stem(w)))
        self.stems = lex.groupby("stem")[["valence_mean", "arousal_mean"]].mean().to_dict("index")
        self._cache: dict[str, dict | None] = {}

    def lookup(self, token: str):
        if token in self._cache:
            return self._cache[token]
        key = strip_accents(token)
        hit = self.exact.get(key)
        if hit is None and len(token) > 3:
            hit = self.stems.get(strip_accents(self.stemmer.stem(token)))
        self._cache[token] = hit
        return hit

    def score(self, text: str) -> tuple[float, float, float]:
        toks = [t for t in tokenize(text) if t not in portuguese_stopwords() and not t.isdigit()]
        hits = [h for h in (self.lookup(t) for t in toks) if h]
        if not hits:
            return np.nan, np.nan, 0.0 if toks else np.nan
        return (float(np.mean([h["valence_mean"] for h in hits])),
                float(np.mean([h["arousal_mean"] for h in hits])),
                len(hits) / len(toks))


class BigramFallbackLM:
    """Bigram model that falls back to an add-one unigram when the bigram was never seen (course unit LM-01)."""

    def __init__(self, texts: list[str], alpha: float = 0.4):
        self.alpha = alpha
        self.uni, self.bi, self.ctx = Counter(), Counter(), Counter()
        for t in texts:
            self.add(t, +1)

    @staticmethod
    def _seq(text: str) -> list[str]:
        return ["<s>"] + tokenize(text) + ["</s>"]

    def add(self, text: str, sign: int):
        seq = self._seq(text)
        for w in seq[1:]:
            self.uni[w] += sign
        for a, b in zip(seq, seq[1:]):
            self.bi[(a, b)] += sign
            self.ctx[a] += sign

    def perplexity(self, text: str) -> float:
        seq = self._seq(text)
        n_tokens = sum(self.uni.values())
        vocab = len([w for w, c in self.uni.items() if c > 0]) + 1
        logp = 0.0
        for a, b in zip(seq, seq[1:]):
            if self.bi[(a, b)] > 0 and self.ctx[a] > 0:
                p = self.bi[(a, b)] / self.ctx[a]
            else:
                p = self.alpha * (self.uni[b] + 1) / (n_tokens + vocab)
            logp += math.log(p)
        return math.exp(-logp / (len(seq) - 1))


def opening_text(row) -> str:
    hook = row.get("hook") if isinstance(row.get("hook"), str) else ""
    if len(tokenize(hook)) >= 3:
        return hook
    return " ".join(tokenize(clean_text(row.get("caption") or ""))[:15])


def build_features(df: pd.DataFrame, anew_csv: Path) -> pd.DataFrame:
    df = df.copy()
    df["caption_clean"] = df["caption"].fillna("").map(clean_text)
    if "transcript" not in df:
        df["transcript"] = ""
    df["transcript"] = df["transcript"].fillna("")
    df["opening"] = df.apply(opening_text, axis=1)
    df["text_all"] = (df["transcript"].str.lower() + " " + df["caption_clean"]).str.strip()

    raw_all = (df["transcript"].fillna("") + " " + df["caption"].fillna("")).str.lower()
    for name, rx in DEVICE_RE.items():
        df[f"dev_{name}"] = raw_all.map(lambda s: int(bool(rx.search(s))))
    df["opening_question"] = df["opening"].map(lambda s: int("?" in s or bool(DEVICE_RE["question"].search(s))))

    anew = Anew(anew_csv)
    for src, prefix in (("opening", "open"), ("text_all", "all")):
        scores = df[src].map(anew.score)
        df[f"{prefix}_valence"] = [s[0] for s in scores]
        df[f"{prefix}_arousal"] = [s[1] for s in scores]
        df[f"{prefix}_anew_coverage"] = [s[2] for s in scores]

    df["n_caption_words"] = df["caption_clean"].map(lambda s: len(tokenize(s)))
    df["n_opening_words"] = df["opening"].map(lambda s: len(tokenize(s)))
    df["n_hashtags"] = df["caption"].fillna("").map(lambda s: len(HASHTAG.findall(s)))
    df["n_emojis"] = df["caption"].fillna("").map(lambda s: len(EMOJI.findall(s)))
    if "n_spoken_words" in df and "audio_seconds" in df:
        df["words_per_second"] = df["n_spoken_words"] / df["audio_seconds"].replace(0, np.nan)

    df["opening_perplexity"] = np.nan
    for user, g in df.groupby("username"):
        train = g[g["split"] == "train"]
        lm = BigramFallbackLM(train["opening"].tolist())
        for idx, row in g.iterrows():
            if row["split"] == "train":
                lm.add(row["opening"], -1)
                df.at[idx, "opening_perplexity"] = lm.perplexity(row["opening"])
                lm.add(row["opening"], +1)
            else:
                df.at[idx, "opening_perplexity"] = lm.perplexity(row["opening"])
    df["log_opening_perplexity"] = np.log(df["opening_perplexity"])
    return df


FEATURE_COLUMNS = ([f"dev_{k}" for k in DEVICES] + ["opening_question", "open_valence", "open_arousal",
                   "all_valence", "all_arousal", "n_caption_words", "n_opening_words", "n_hashtags", "n_emojis",
                   "log_opening_perplexity"])
