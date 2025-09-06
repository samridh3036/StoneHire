import faiss
import numpy as np
from typing import List, Dict, Tuple

class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors for cosine
        self.index = index
        self.id_map = []  # keep order -> id
        self.vectors = None

    def add(self, ids: List[str], vecs: np.ndarray):
        # normalize for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms==0]=1
        vecs = vecs / norms
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.id_map.extend(ids)

    def search(self, qvec: np.ndarray, topk=50):
        faiss.normalize_L2(qvec)
        D, I = self.index.search(qvec, topk)
        results = []
        for dist_row, idx_row in zip(D, I):
            for score, idx in zip(dist_row, idx_row):
                if idx == -1:
                    continue
                results.append({"id": self.id_map[idx], "score": float(score)})
        return results
