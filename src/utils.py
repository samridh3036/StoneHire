import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile
import email
import datetime

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[\s-])?(?:\d{10}|\d{3}[\s-]\d{3}[\s-]\d{4})")

SECTION_HEADINGS = [
    "EDUCATION",
    "INTERNSHIPS",
    "INTERNSHIPS and PROJECTS",
    "PROJECTS",
    "COMPETITION",
    "AWARDS",
    "SKILLS AND EXPERTISE",
    "POSITIONS OF RESPONSIBILITY",
    "EXTRA CURRICULAR ACTIVITIES"
]

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def unzip_to_dir(zip_path: str, out_dir: str) -> List[str]:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    # return list of extracted file paths
    files=[]
    for root, _, filenames in os.walk(out_dir):
        for fn in filenames:
            files.append(os.path.join(root, fn))
    return files

def simple_contact_info(text: str) -> Dict:
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    name = None
    # heuristics: first non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        name = lines[0]
    return {"name": name, "emails": emails, "phones": phones}

import re

def split_sections(text, headings):
    """
    Splits the resume text into sections based on given headings.
    Returns a dictionary: {heading_name: section_text}.
    """
    sections = {}
    current_heading = None
    buffer = []

    # Create a regex pattern for the headings (case-insensitive)
    pattern = re.compile(r"^\s*(%s)\s*$" % "|".join(map(re.escape, headings)), re.IGNORECASE)

    for line in text.splitlines():
        line_stripped = line.strip()

        # If this line matches a heading, store the previous buffer and start a new section
        if pattern.match(line_stripped):
            if current_heading and buffer:
                sections[current_heading] = "\n".join(buffer).strip()
                buffer = []
            current_heading = line_stripped.upper()
        else:
            buffer.append(line_stripped)

    # Save the last section
    if current_heading and buffer:
        sections[current_heading] = "\n".join(buffer).strip()

    return sections

def normalize_skill(skill: str) -> str:
    return skill.strip().lower()

def safe_read_text_file(path: str) -> str:
    with open(path, 'rb') as f:
        try:
            raw = f.read().decode('utf-8')
            return raw
        except:
            try:
                return open(path, encoding='latin-1').read()
            except:
                return ""

def now_timestamp() -> str:
    return datetime.datetime.utcnow().isoformat()
