from typing import Dict
from ..utils import normalize_skill

SKILL_MAPPING = {
    # small example; expand in production
    "pytorch": "pytorch",
    "py torch": "pytorch",
    "machine learning": "machine learning",
    "ml": "machine learning",
}

def preprocess(parsed: Dict) -> Dict:
    # normalize skills via mapping
    normalized = []
    for s in parsed.get("skills", []):
        normalized.append(SKILL_MAPPING.get(s, s))
    parsed["skills_normalized"] = list(dict.fromkeys(normalized))  # unique preserve order
    # infer seniority
    years = parsed.get("estimated_years_experience") or 0
    if years >= 8:
        parsed["seniority"] = "lead"
    elif years >= 4:
        parsed["seniority"] = "senior"
    elif years >= 2:
        parsed["seniority"] = "mid"
    else:
        parsed["seniority"] = "junior"
    return parsed
