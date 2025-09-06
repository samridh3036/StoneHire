from sentence_transformers import CrossEncoder
from typing import List, Dict
import numpy as np

class CrossReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            print("[reranker] cross-encoder load failed:", e)
            self.model = None

    def score_pairs(self, jd_text: str, resumes: List[Dict]) -> List[float]:
        """
        resumes: list of dicts with 'cand_long' fields
        returns list of scores normalized 0..1
        """
        pairs = [[jd_text, r.get("cand_long","")] for r in resumes]
        if self.model:
            raw = self.model.predict(pairs)
            # normalize to 0..1
            arr = np.array(raw, dtype=float)
            if arr.max()==arr.min():
                return [0.5]*len(arr)
            arr = (arr - arr.min()) / (arr.max()-arr.min())
            return arr.tolist()
        else:
            # fallback: coarse heuristic using overlap of skills
            scores=[]
            for r in resumes:
                skills = set(r.get("skills_normalized",[]))
                match = len(skills)
                scores.append(min(1.0, match/10))
            return scores
