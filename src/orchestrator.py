import yaml
from pathlib import Path
from src.agents.ingest import ingest
from src.agents.parser import parse_resume
from src.agents.preprocess import preprocess
from src.agents.embedder import Embedder, build_embedding_entries
from src.agents.retriever import FaissIndex
from src.agents.reranker import CrossReranker
from src.agents.fairness import apply_simple_fairness
from src.agents.critic import critic_review
import numpy as np
import os, json, argparse

CONFIG_PATH = Path("./config.yaml")

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def read_jd(jd_file: str = None, jd_text: str = None) -> str:
    if jd_file:
        return Path(jd_file).read_text()
    if jd_text:
        return jd_text
    # fallback: ask user interactively
    print("Paste the job description, then CTRL+D (or enter EOF):")
    return "".join(iter(input, ""))  # not ideal in some consoles

def run_pipeline(zip_path: str = None, jd_file: str = None, jd_text: str = None):
    cfg = load_config()
    zip_path = zip_path or cfg["data"]["zip_path"]
    extracted_dir = cfg["data"]["extracted_dir"]
    top_k = cfg["processing"]["top_k_retrieve"]
    final_k = cfg["processing"]["final_top_k"]

    # 1. Ingest
    print("[orchestrator] ingesting...")
    docs = ingest(zip_path, extracted_dir)
    print(f"[orchestrator] {len(docs)} files extracted")

    # 2. Parse & preprocess
    parsed=[]
    for d in docs:
        p = parse_resume(d)
        p = preprocess(p)
        parsed.append(p)

    if not parsed:
        print("[orchestrator] No parsed resumes found. Exiting.")
        return []

    # 3. Embedding
    embedder = Embedder(cfg["processing"]["embedding_model"])
    entries = build_embedding_entries(parsed, embedder)
    vectors = np.stack([e["vector"] for e in entries])
    ids = [e["id"] for e in entries]

    # 4. Build FAISS index
    dim = vectors.shape[1]
    fi = FaissIndex(dim)
    fi.add(ids, vectors)

    # 5. JD input
    jd_text_final = read_jd(jd_file, jd_text)
    if not jd_text_final or len(jd_text_final.strip())==0:
        print("[orchestrator] Empty JD provided. Exiting.")
        return []

    # 6. Query embed
    qvec = embedder.model.encode([jd_text_final], convert_to_numpy=True).astype("float32")
    raw_hits = fi.search(qvec, topk=top_k)
    if not raw_hits:
        print("[orchestrator] No hits returned from retriever.")
        return []

    hit_ids = [h["id"] for h in raw_hits]

    # 7. gather candidate docs in same order
    id_to_doc = {p["id"]: p for p in parsed}
    candidates = [id_to_doc[h] for h in hit_ids if h in id_to_doc]

    # 8. rerank with cross-encoder
    reranker = CrossReranker(cfg["processing"]["cross_encoder_model"])
    c_scores = reranker.score_pairs(jd_text_final, candidates)

    # 9. deterministic features score (example: skill overlap)
    feature_scores = []
    for c in candidates:
        feature_scores.append(min(1.0, len(c.get("skills_normalized",[]))/5.0))

    # 10. combine
    w_e = cfg["scoring"]["w_e"]
    w_c = cfg["scoring"]["w_c"]
    w_f = cfg["scoring"]["w_f"]

    emb_score_map = {h["id"]: h["score"] for h in raw_hits}
    final_list=[]
    for i, c in enumerate(candidates):
        eid = c["id"]
        E = emb_score_map.get(eid, 0.0)
        C = c_scores[i]
        F = feature_scores[i]
        composite = w_e*E + w_c*C + w_f*F
        final_list.append({"id": eid, "candidate": c, "embedding_score": float(E), "cross_score": float(C), "feature_score": float(F), "final_score": float(composite)})

    # 11. fairness pass (optional) - requires metadata.university to exist; safe if missing
    final_list = critic_review(final_list, jd_text_final)

    # 12. fairness pass (optional)
    if cfg.get("fairness", {}).get("enabled", False):
        final_list = apply_simple_fairness(final_list, protected_field="university", max_share=0.5)

    # 12. top-K output (only roll numbers)
    final_sorted = sorted(final_list, key=lambda x: x["final_score"], reverse=True)[:final_k]

    # Ensure results dir
    Path("results").mkdir(exist_ok=True)

    # Write simple roll number list (one per line)
    roll_list = [item["id"] for item in final_sorted]
    Path("results/top_roll_numbers.txt").write_text("\n".join(roll_list))
    # Also write full JSON for debug/optional use
    out_json = []
    for item in final_sorted:
        c = item["candidate"]
        out_json.append({
            "roll_number": item["id"],
            "filename": c.get("filename"),
            "score": item["final_score"],
            "summary": item.get("fit_summary", ""),
        })

    Path("results/top_candidates.json").write_text(json.dumps(out_json, indent=2))

    print(f"[orchestrator] Done — top {len(roll_list)} roll numbers saved to results/top_roll_numbers.txt")
    return roll_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run resume ranking pipeline. Resumes should be named by roll numbers.")
    parser.add_argument("--zip_path", type=str, help="Path to resumes zip (overrides config)", default=None)
    parser.add_argument("--jd_file", type=str, help="Path to job description txt file", default=None)
    parser.add_argument("--jd_text", type=str, help="Provide JD as a string (be careful with shell quoting)", default=None)
    args = parser.parse_args()
    run_pipeline(zip_path=args.zip_path, jd_file=args.jd_file, jd_text=args.jd_text)
