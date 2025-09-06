from typing import Dict, List
def explain_candidate(jd_text: str, candidate: Dict) -> Dict:
    # produce a short rationale (deterministic): skills overlap + years + top matched snippets
    req_skills = set()  # in production, parse JD to extract req skills
    cand_skills = set(candidate.get("skills_normalized",[]))
    overlap = list(req_skills & cand_skills)
    rationale = f"Matched skills: {', '.join(overlap) if overlap else 'N/A'}. Estimated years: {candidate.get('estimated_years_experience')}"
    return {"id": candidate["id"], "rationale": rationale}
