# src/agents/critic.py
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from transformers import pipeline, AutoTokenizer

# ---------- CONFIG ----------
MODEL_NAME = "facebook/bart-large-cnn"
# BART encoder+decoder max position embeddings ~1024 tokens. Keep a safe margin.
BART_MAX_TOKENS = 1024
SAFE_MARGIN_TOKENS = 80   # reserve for prompt overhead & safety
# Default chunk size used if prompt length estimation not possible
DEFAULT_CHUNK_TOKENS = 700

# How many candidates processed in parallel (threads)
MAX_CANDIDATE_WORKERS = max(1, (os.cpu_count() or 2) - 1)
# How many agent calls inside a candidate can be parallelized
MAX_AGENT_WORKERS = 3

# ---------- MODEL LOAD (global) ----------
# Load tokenizer + pipeline once per process/thread
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
summarizer = pipeline("text2text-generation", model=MODEL_NAME, device=-1)


# ---------- UTIL: token-aware chunking ----------
def chunk_text_by_prompt_capacity(text: str, extra_prompt_text: str = "", max_input_tokens: int = BART_MAX_TOKENS) -> List[str]:
    """
    Splits `text` into chunks such that each chunk + extra_prompt_text will fit under
    max_input_tokens - SAFE_MARGIN_TOKENS. We use tokenizer to count tokens.
    """
    if not text:
        return []

    # token counts
    extra_tokens = len(tokenizer.encode(extra_prompt_text, truncation=False))
    capacity = max_input_tokens - extra_tokens - SAFE_MARGIN_TOKENS
    if capacity <= 50:
        capacity = DEFAULT_CHUNK_TOKENS  # fallback

    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + capacity]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        i += capacity
    return chunks


# ---------- UTIL: JSON cleaning & parsing ----------
def extract_json_like(text: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object from model output and parse it.
    Returns dict if ok, otherwise returns {'_raw': cleaned_text}
    """
    if not text:
        return {}

    # try to find a {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    candidate = m.group(0) if m else text

    # normalize smart quotes etc
    candidate = candidate.replace("’", "'").replace("“", '"').replace("”", '"')
    # replace lone single quotes with double quotes for JSON
    candidate = re.sub(r"(?<!\\)'", '"', candidate)

    # minimal cleanup: remove repeated "Begin JSON output" text
    candidate = re.sub(r"Begin JSON output below[:\s]*", "", candidate, flags=re.I)

    # try to decode
    try:
        parsed = json.loads(candidate)
        return parsed
    except Exception:
        # try more lenient fixes: remove trailing commas
        candidate2 = re.sub(r",\s*([\]\}])", r"\1", candidate)
        try:
            parsed = json.loads(candidate2)
            return parsed
        except Exception:
            # fallback: return raw string so you still have content
            return {"_raw": candidate.strip()}


# ---------- Phase 1: Extract facts (single function) ----------
def extract_resume_facts(full_resume_text: str) -> str:
    """
    Extract high-signal facts from a resume in concise bullet points.
    Returns a single string summarizing extracted facts (structured bullets).
    """
    if not full_resume_text:
        return ""

    # The prompt for extraction is short and focused
    extractor_prompt_header = (
        "Extract from the resume ONLY these structured facts:\n"
        "- TECHNICAL_SKILLS: comma-separated concise tokens (libraries/tools)\n"
        "- EDUCATION: one-line degree & major\n"
        "- EXPERIENCE_YEARS: numeric if present\n"
        "- PROJECTS: short 'Title — outcome' bullets (2-3 items)\n"
        "- ACHIEVEMENTS: short bullets (awards, leadership)\n\n"
        "Return a JSON object exactly with these keys."
    )

    # Create chunks using prompt-capacity so extractor prompt + chunk fit
    chunks = chunk_text_by_prompt_capacity(full_resume_text, extra_prompt_text=extractor_prompt_header)
    extracted_parts = []
    for chunk in chunks:
        prompt = f"{extractor_prompt_header}\n\nResume chunk:\n{chunk}"
        out = summarizer(prompt, max_length=220, min_length=40, do_sample=False)[0]["generated_text"]
        parsed = extract_json_like(out)
        # keep stringified parsed (prefer JSON fields if available)
        if isinstance(parsed, dict) and parsed:
            extracted_parts.append(parsed)
        else:
            # fallback: keep raw output
            extracted_parts.append({"_raw": out.strip()})

    # Merge extracted_parts into a consolidated structured string
    # For practicality, convert to a simple readable summary string for agents
    merged = {
        "TECHNICAL_SKILLS": set(),
        "EDUCATION": [],
        "EXPERIENCE_YEARS": None,
        "PROJECTS": [],
        "ACHIEVEMENTS": []
    }
    for part in extracted_parts:
        if "_raw" in part and len(part) == 1:
            # keep as extra note
            merged.setdefault("_raw_notes", []).append(part["_raw"])
            continue
        # TECHNICAL_SKILLS
        skills = part.get("TECHNICAL_SKILLS")
        if isinstance(skills, str):
            for s in re.split(r"[,\n;|]", skills):
                s = s.strip()
                if s:
                    merged["TECHNICAL_SKILLS"].add(s)
        elif isinstance(skills, list):
            for s in skills:
                merged["TECHNICAL_SKILLS"].add(s.strip())
        # EDUCATION
        edu = part.get("EDUCATION")
        if edu:
            merged["EDUCATION"].append(edu.strip())
        # EXPERIENCE_YEARS
        yrs = part.get("EXPERIENCE_YEARS")
        if yrs and not merged["EXPERIENCE_YEARS"]:
            try:
                merged["EXPERIENCE_YEARS"] = int(re.findall(r"\d+", str(yrs))[0])
            except Exception:
                pass
        # PROJECTS
        projects = part.get("PROJECTS")
        if isinstance(projects, list):
            merged["PROJECTS"].extend([p.strip() for p in projects if p])
        elif isinstance(projects, str):
            for p in re.split(r"\n|•|-", projects):
                p = p.strip()
                if p:
                    merged["PROJECTS"].append(p)
        # ACHIEVEMENTS
        ach = part.get("ACHIEVEMENTS")
        if isinstance(ach, list):
            merged["ACHIEVEMENTS"].extend([a.strip() for a in ach if a])
        elif isinstance(ach, str):
            for a in re.split(r"\n|•|-", ach):
                a = a.strip()
                if a:
                    merged["ACHIEVEMENTS"].append(a)

    # produce readable structured string
    lines = []
    if merged["TECHNICAL_SKILLS"]:
        lines.append("TECHNICAL_SKILLS: " + ", ".join(sorted(merged["TECHNICAL_SKILLS"])))
    if merged["EDUCATION"]:
        lines.append("EDUCATION: " + " | ".join(merged["EDUCATION"]))
    if merged["EXPERIENCE_YEARS"]:
        lines.append(f"EXPERIENCE_YEARS: {merged['EXPERIENCE_YEARS']}")
    if merged["PROJECTS"]:
        lines.append("PROJECTS: " + " || ".join(merged["PROJECTS"][:4]))
    if merged["ACHIEVEMENTS"]:
        lines.append("ACHIEVEMENTS: " + " || ".join(merged["ACHIEVEMENTS"][:4]))
    if "_raw_notes" in merged:
        lines.append("NOTES: " + " | ".join(merged["_raw_notes"][:2]))

    return "\n".join(lines)


# ---------- Agent templates ----------
def run_technical_agent(structured_resume: str, jd_text: str, final_score: float) -> Dict[str, Any]:
    prompt = (
        "You are a technical hiring reviewer. Using the candidate's extracted resume facts below, "
        "produce JSON with keys: Tech_Strengths (3 items), Tech_Gaps (2 items), "
        "Tech_Assessment (1-2 sentences connecting specific resume evidence to JD), "
        "Score_Defense (one sentence explaining the numeric score provided).\n\n"
        f"JOB:\n{jd_text}\n\n"
        f"EXTRACTED_FACTS:\n{structured_resume}\n\n"
        f"Given pipeline_score: {final_score:.4f}\n\n"
        "Return exactly one JSON object and nothing else."
    )
    out = summarizer(prompt, max_length=220, min_length=80, do_sample=False)[0]["generated_text"]
    return extract_json_like(out)


def run_softskills_agent(structured_resume: str, jd_text: str, final_score: float) -> Dict[str, Any]:
    prompt = (
        "You are a soft-skills reviewer. From the extracted facts, return JSON with keys: "
        "Soft_Strengths (2-3 items), Soft_Gaps (1-2 items), Soft_Assessment (1-2 sentences), "
        "Score_Defense (one sentence linking score to soft-skill evidence).\n\n"
        f"JOB:\n{jd_text}\n\n"
        f"EXTRACTED_FACTS:\n{structured_resume}\n\n"
        f"Given pipeline_score: {final_score:.4f}\n\n"
        "Return exactly one JSON object and nothing else."
    )
    out = summarizer(prompt, max_length=200, min_length=60, do_sample=False)[0]["generated_text"]
    return extract_json_like(out)


def run_leadership_agent(structured_resume: str, jd_text: str, final_score: float) -> Dict[str, Any]:
    prompt = (
        "You are a leadership/impact reviewer. Return JSON with keys: "
        "Lead_Strengths (1-2 items), Lead_Risks (1-2 items), Lead_Assessment (1-2 sentences), "
        "Score_Defense (one sentence linking score to leadership/impact evidence).\n\n"
        f"JOB:\n{jd_text}\n\n"
        f"EXTRACTED_FACTS:\n{structured_resume}\n\n"
        f"Given pipeline_score: {final_score:.4f}\n\n"
        "Return exactly one JSON object and nothing else."
    )
    out = summarizer(prompt, max_length=180, min_length=50, do_sample=False)[0]["generated_text"]
    return extract_json_like(out)


def run_aggregator_agent(tech_json: Dict, soft_json: Dict, lead_json: Dict, final_score: float, jd_text: str) -> Dict[str, Any]:
    """
    Merge agent outputs into final verdict JSON.
    """
    prompt = (
        "You are the lead aggregator. Merge these reviewer JSONs into one final JSON with keys:\n"
        "Verdict (Hire|Maybe|Reject with %), Overall_Strengths (3 items), Overall_Weaknesses (2 items), "
        "Match_Reasoning (2-3 sentences with evidence), Score_Defense (one sentence combining the three agents), Punchline (1 short recruiter-style line).\n\n"
        f"JOB:\n{jd_text}\n\n"
        f"TECH_JSON:\n{json.dumps(tech_json)}\n\n"
        f"SOFT_JSON:\n{json.dumps(soft_json)}\n\n"
        f"LEAD_JSON:\n{json.dumps(lead_json)}\n\n"
        f"Pipeline score: {final_score:.4f}\n\n"
        "Return exactly one JSON object and nothing else."
    )
    out = summarizer(prompt, max_length=320, min_length=120, do_sample=False)[0]["generated_text"]
    return extract_json_like(out)


# ---------- Worker for a single candidate ----------
def process_candidate_multi_agent(candidate_entry: Dict, jd_text: str, company_name: str = "Our Company", role: str = "AI/ML Engineer") -> Dict:
    """
    candidate_entry is expected to be a dict with keys:
      - id or roll number accessible via candidate_entry.get('id') or entry['id']
      - candidate: parsed candidate dict (from parser), including text sections
      - final_score: numeric score the pipeline computed (float)
    Returns the same entry augmented with 'fit_summary' (dict) and 'fit_json' etc.
    """
    # Pull fields
    parsed_cand = candidate_entry.get("candidate", {})
    # Candidate full text: prefer full_text, cand_long, or join sections
    resume_text = parsed_cand.get("full_text") or parsed_cand.get("cand_long") or "\n".join(
        [f"{k}: {v}" for k, v in parsed_cand.get("sections", {}).items() if isinstance(v, str)]
    )
    final_score = float(candidate_entry.get("final_score", candidate_entry.get("score", 0.0)))

    # Phase 1: Extract facts
    structured_resume = extract_resume_facts(resume_text)

    # Phase 2: Run agents (parallel inside candidate)
    agent_results = {}
    with ThreadPoolExecutor(max_workers=MAX_AGENT_WORKERS) as aexec:
        futures = {
            aexec.submit(run_technical_agent, structured_resume, jd_text, final_score): "tech",
            aexec.submit(run_softskills_agent, structured_resume, jd_text, final_score): "soft",
            aexec.submit(run_leadership_agent, structured_resume, jd_text, final_score): "lead"
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                agent_results[key] = fut.result()
            except Exception as e:
                agent_results[key] = {"_error": str(e)}

    # Phase 3: Aggregator
    agg = run_aggregator_agent(agent_results.get("tech", {}), agent_results.get("soft", {}), agent_results.get("lead", {}), final_score, jd_text)

    # Attach structured info to candidate_entry
    candidate_entry["critic"] = {
        "structured_resume": structured_resume,
        "technical_agent": agent_results.get("tech"),
        "softskills_agent": agent_results.get("soft"),
        "leadership_agent": agent_results.get("lead"),
        "aggregator": agg
    }

    return candidate_entry


# ---------- Top-level multi-threaded critic entrypoint ----------
def critic_review(
    candidates: List[Dict], 
    jd_text: str, 
    company_name: str = "Our Company", 
    role: str = "AI/ML Engineer", 
    max_workers: int = MAX_CANDIDATE_WORKERS
) -> List[Dict]:
    """
    Multi-agent, multi-threaded critic. Processes candidate list in parallel.
    Each candidate gets a `critic` dict attached with agent outputs and final aggregator.
    Outer pool is capped for predictable concurrency.
    """
    # Safety: at least 1 worker
    max_workers = max(1, max_workers)

    reviewed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_candidate_multi_agent, c, jd_text, company_name, role): c
            for c in candidates
        }
        for fut in as_completed(futures):
            try:
                reviewed.append(fut.result())
            except Exception as e:
                candidate_id = futures[fut].get("roll_number", "unknown")
                reviewed.append({
                    "roll_number": candidate_id,
                    "_error": str(e)
                })

    return reviewed
