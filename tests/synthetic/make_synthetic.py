"""Synthetic dataset in the same raw format the collection snippet exports, with planted word effects.

10 creators (2 per sector), 110 reels each. Views follow account growth plus planted effects:
sector words that help (+), sector words that hurt (-), and generic devices (question opening, "pare de").
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
COLLECTED = datetime(2026, 11, 23, 15, 0, tzinfo=timezone.utc)

SECTORS = {
    "ai": {"topic": ["chatgpt", "prompt", "automação", "ferramenta", "agente", "modelo", "planilha", "produtividade"],
           "good": ["segredo", "grátis"], "bad": ["atualização", "notícia"]},
    "hair_visagism": {"topic": ["corte", "rosto", "franja", "degradê", "barba", "visagismo", "formato", "cabelo"],
                      "good": ["transformação", "antes"], "bad": ["produto", "promoção"]},
    "consulting": {"topic": ["empresa", "processo", "gestão", "equipe", "cliente", "faturamento", "margem", "líder"],
                   "good": ["erro", "lucro"], "bad": ["evento", "parceria"]},
    "trading": {"topic": ["gráfico", "candle", "índice", "dólar", "stop", "operação", "tendência", "suporte"],
                "good": ["perdi", "estratégia"], "bad": ["live", "sorteio"]},
    "sports": {"topic": ["jogo", "time", "gol", "técnico", "campeonato", "treino", "torcida", "atleta"],
               "good": ["polêmica", "absurdo"], "bad": ["resumo", "agenda"]},
}
FILLER = ["hoje eu vou te mostrar", "presta atenção nisso", "isso muda tudo", "vamos falar sobre",
          "olha só o que aconteceu", "muita gente não sabe", "fica até o final", "isso aqui é importante"]
HASHTAGS = ["#dicas", "#brasil", "#reels", "#conteudo", "#viral"]


def make_creator(rng, user, sector, cfg, n=110):
    items, transcripts = [], []
    start = COLLECTED - timedelta(days=3 * n)
    base = rng.uniform(np.log(2e4), np.log(3e5))
    for i in range(n):
        posted = start + timedelta(days=3 * i, hours=float(rng.uniform(0, 12)))
        words = list(rng.choice(cfg["topic"], 4))
        effect = 0.0
        if rng.random() < 0.3:
            words.append(rng.choice(cfg["good"])); effect += 0.55
        if rng.random() < 0.3:
            words.append(rng.choice(cfg["bad"])); effect -= 0.55
        question = rng.random() < 0.35
        prohibition = rng.random() < 0.25
        cta_comment = rng.random() < 0.3
        hook = ("você sabia que " if question else "") + ("pare de " if prohibition else "") + " ".join(words[:3])
        if question:
            hook += "?"
            effect += 0.25
        if prohibition:
            effect += 0.2
        body = f"{rng.choice(FILLER)} {' '.join(words[2:])} {rng.choice(FILLER)} {rng.choice(cfg['topic'])}"
        transcript = f"{hook} {body}" + (" comenta eu quero aqui embaixo" if cta_comment else "")
        caption = (f"{' '.join(words[:2])} {rng.choice(FILLER)} 🔥 " + " ".join(rng.choice(HASHTAGS, 2))
                   + (" comenta EU QUERO" if cta_comment else ""))
        growth = 0.6 * i / n
        views = float(np.exp(base + growth + effect + rng.normal(0, 0.55)))
        if (COLLECTED - posted).days < 14:
            views *= 0.4
        comments = int(views * 0.002 * np.exp((0.9 if cta_comment else 0) + rng.normal(0, 0.4)))
        code = f"{user}_{i:04d}"
        items.append({
            "code": code, "pk": str(10**12 + hash(code) % 10**9), "taken_at": int(posted.timestamp()),
            "media_type": 2, "product_type": "clips", "play_count": int(views), "like_count": int(views * 0.04),
            "comment_count": comments, "video_duration": float(rng.uniform(15, 60)), "caption": {"text": caption},
            "video_versions": [{"url": "https://example.invalid/video.mp4"}],
            "user": {"username": user, "follower_count": int(np.exp(base) * 3)},
            "coauthor_producers": [], "is_paid_partnership": bool(rng.random() < 0.05),
        })
        transcripts.append({"reel_id": code, "transcript": transcript, "hook": hook,
                            "n_spoken_words": len(transcript.split()), "speech_seconds": 30.0, "audio_seconds": 35.0})
    return items, transcripts


def main(seed=7):
    rng = np.random.default_rng(seed)
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    creators, all_tr = [], []
    for sector, cfg in SECTORS.items():
        for k in (1, 2):
            user = f"{sector}_creator{k}"
            items, tr = make_creator(rng, user, sector, cfg)
            payload = {"username": user, "collected_at": COLLECTED.isoformat(), "items": items}
            (RAW / f"{user}_reels_2026-11-23.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            creators.append({"username": user, "sector": sector})
            all_tr += tr
    pd.DataFrame(creators).to_csv(ROOT / "data" / "creators.csv", index=False)
    pd.DataFrame(all_tr).to_parquet(PROC / "transcripts.parquet", index=False)
    print(f"wrote {len(creators)} creators, {len(all_tr)} reels")


if __name__ == "__main__":
    main()
