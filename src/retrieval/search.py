from typing import List, Dict, Tuple
from functools import lru_cache
import os, numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.index_faiss import load_index

TOP_K = int(os.getenv("TOP_K", "6"))

@lru_cache(maxsize=1)
def _load_meta_corpus() -> Tuple[List[Dict], BM25Okapi]:
    _, meta = load_index()
    tokenized = [m["text"].lower().split() for m in meta]
    bm25 = BM25Okapi(tokenized)
    return meta, bm25

def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def hybrid_search(query: str) -> List[Dict]:
    meta, bm25 = _load_meta_corpus()

    bm25_scores_all = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(bm25_scores_all)[::-1][:TOP_K]
    bnorm = _normalize(bm25_scores_all[bm25_top])

    results = []
    for rank, idx in enumerate(bm25_top):
        m = meta[idx]
        score = float(bnorm[rank])
        results.append({
            "text": m["text"],
            "source": m.get("source", f"policy://{m.get('policy_id','unknown')}/{m.get('section','')}"),
            "score": score,
            "policy_id": m.get("policy_id"),
            "section": m.get("section"),
            "effective_from": m.get("effective_from"),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:TOP_K]
