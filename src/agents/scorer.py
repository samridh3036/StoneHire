import re
from collections import defaultdict

# Extract keywords from JD
def extract_jd_keywords(jd_text):
    words = re.findall(r"\b[a-zA-Z]{2,}\b", jd_text.lower())
    stopwords = set(["and", "or", "with", "the", "in", "for", "on", "at", "to", "of", "a", "an"])
    keywords = [w for w in words if w not in stopwords]
    return list(set(keywords))

# Score the resume against JD
def score_resume(parsed_resume, jd_keywords):
    # Weights for different sections
    section_weights = {
        "SKILLS AND EXPERTISE": 0.4,
        "PROJECTS": 0.2,
        "INTERNSHIPS": 0.1,
        "INTERNSHIPS AND PROJECTS": 0.1,
        "EXPERIENCE": 0.1,
        "CERTIFICATION": 0.05,
        "EDUCATION": 0.05
    }

    matched_keywords = defaultdict(list)
    total_score = 0

    for section, weight in section_weights.items():
        section_text = parsed_resume["sections"].get(section, "").lower()
        if not section_text:
            continue
        matches = [kw for kw in jd_keywords if kw in section_text]
        matched_keywords[section] = matches
        if matches:
            total_score += (len(matches) / len(jd_keywords)) * weight

    return total_score, matched_keywords

# Generate summary
def generate_summary(parsed_resume, matched_keywords):
    summary_parts = []

    # Education
    edu = parsed_resume["sections"].get("EDUCATION", "").strip()
    if edu:
        summary_parts.append(f"Education: {edu.splitlines()[0]}")

    # Experience & Projects
    if parsed_resume["sections"].get("EXPERIENCE") or parsed_resume["sections"].get("PROJECTS"):
        summary_parts.append("Has relevant project and/or work experience related to the JD.")

    # Certifications
    cert = parsed_resume["sections"].get("CERTIFICATION", "")
    if cert:
        summary_parts.append("Has certifications that strengthen the JD match.")

    # Skills
    skills = matched_keywords.get("SKILLS AND EXPERTISE", [])
    if skills:
        summary_parts.append("Key skills match: " + ", ".join(skills[:10]))

    # Special note if LLM/AI-related terms found
    all_matches = sum(matched_keywords.values(), [])
    if any(kw in ["llm", "ai", "artificial intelligence", "machine learning"] for kw in all_matches):
        summary_parts.append("Has direct AI/ML/LLM experience relevant to the role.")

    return " ".join(summary_parts)
