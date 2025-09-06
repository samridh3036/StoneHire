from typing import Dict
from ..utils import split_sections, simple_contact_info, normalize_skill
import re

# List of section headings to detect
SECTION_HEADINGS = [
    "EDUCATION",
    "INTERNSHIPS",
    "PROJECTS",
    "INTERNSHIPS AND PROJECTS",
    "ACADEMIC ACHIEVEMENT",
    "CERTIFICATION",
    "TRAINING",
    "EXPERIENCE",
    "ENTREPRENEURIAL EXPERIENCE",
    "COMPETITION/CONFERENCE",
    "PUBLICATION",
    "POSITION OF RESPONSIBILITIES",
    "EXTRA-CURRICULAR ACTIVITIES",
    "SKILLS AND EXPERTISE",
    "COURSEWORK INFORMATION"
]


def parse_resume(doc: Dict) -> Dict:
    text = doc.get("raw_text", "")
    
    # Split the resume into sections based on the above headings
    sections = split_sections(text, headings=SECTION_HEADINGS)
    
    # Extract contact info
    contact = simple_contact_info(text)
    
    # Gather skills from multiple possible sources
    skills_text = ""
    for heading in ["SKILLS AND EXPERTISE", "TECHNICAL SKILLS"]:
        if heading in sections:
            skills_text += "\n" + sections[heading]
    
    # If still empty, try to detect skill-like lines anywhere
    if not skills_text:
        candidates = []
        for ln in text.splitlines():
            if any(sep in ln for sep in [",", "|", ";"]) and len(ln) < 200 and re.search(r"(skills|technologies|tools|frameworks)", ln, re.I):
                candidates.append(ln)
        skills_text = "\n".join(candidates)
    
    # Normalize skills
    skills = []
    for part in re.split(r"[,;\n•\u2022]", skills_text):
        s = part.strip()
        if len(s) > 1:
            skills.append(normalize_skill(s))
    
    # Estimate experience (naive year count)
    years = None
    m = re.search(r"(\d+)\s+(?:years|yrs)\b", text, re.IGNORECASE)
    if m:
        years = int(m.group(1))
    
    # Create parsed resume dict
    parsed = {
        "id": doc["id"],
        "filename": doc["filename"],
        "name": contact.get("name"),
        "emails": contact.get("emails"),
        "phones": contact.get("phones"),
        "sections": sections,  # all sections stored here
        "skills": skills,
        "estimated_years_experience": years
    }
    
    # Short representation
    short = []
    if parsed["name"]:
        short.append(parsed["name"])
    if skills:
        short.append(", ".join(skills[:12]))
    if parsed["estimated_years_experience"]:
        short.append(f"{parsed['estimated_years_experience']} years exp")
    parsed["cand_short"] = " • ".join(short)
    
    # Full text
    parsed["cand_long"] = text
    
    return parsed
