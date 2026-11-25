"""Frozen BERTimbau [CLS] embeddings (course LM-07/LM-08 BertClassifier with freeze_bert=True, without the linear head)."""
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"


def embed_texts(texts: pd.Series, cache: Path, max_length: int = 128, batch_size: int = 32) -> np.ndarray:
    if cache.exists():
        stored = pd.read_parquet(cache)
        if stored.index.equals(texts.index):
            return stored.to_numpy()
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    out = []
    with torch.no_grad():
        values = texts.fillna("").tolist()
        for i in range(0, len(values), batch_size):
            enc = tok(values[i:i + batch_size], truncation=True, padding=True, max_length=max_length,
                      return_tensors="pt")
            out.append(model(**enc).last_hidden_state[:, 0, :].numpy())
    emb = np.vstack(out)
    pd.DataFrame(emb, index=texts.index, columns=[f"e{i}" for i in range(emb.shape[1])]).to_parquet(cache)
    return emb
