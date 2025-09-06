from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class Embedder:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

# simple helper to build doc entries for DB
def build_embedding_entries(parsed_docs: List[Dict], embedder: Embedder):
    ids = [d["id"] for d in parsed_docs]
    texts = [d.get("cand_short","") for d in parsed_docs]
    vecs = embedder.embed_texts(texts).astype("float32")
    entries = []
    for i, d in enumerate(parsed_docs):
        entries.append({
            "id": d["id"],
            "vector": vecs[i],
            "metadata": {
                "filename": d.get("filename"),
                "skills": d.get("skills_normalized"),
                "seniority": d.get("seniority")
            }
        })
    return entries
