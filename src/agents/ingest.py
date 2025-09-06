import os
from pathlib import Path
from ..utils import unzip_to_dir, safe_read_text_file, ensure_dir
from typing import List, Dict
from pdfminer.high_level import extract_text as pdf_extract_text
import docx2txt

TEXT_EXT = {".txt", ".md"}
DOCX_EXT = {".docx"}
PDF_EXT = {".pdf"}

def extract_text_from_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    try:
        if ext in TEXT_EXT:
            return safe_read_text_file(path)
        if ext in DOCX_EXT:
            return docx2txt.process(path) or ""
        if ext in PDF_EXT:
            return pdf_extract_text(path) or ""
    except Exception as e:
        print(f"[ingest] failed to extract {path}: {e}")
        return ""
    return ""

def filename_to_roll(fn: str) -> str:
    """
    Convert a filename like '22CH10059.pdf' or '21EX30027 - resume.pdf' -> '22CH10059'
    Uses first token that matches a roll-like pattern: digits + letters + digits,
    or falls back to basename without extension.
    """
    base = Path(fn).stem  # name without extension
    # common patterns: allow underscores, spaces, hyphens -> split and pick first token that looks like roll
    tokens = [t.strip() for t in re.split(r"[_\-\s]+", base) if t.strip()]
    for t in tokens:
        # pretty permissive: start with digits (2-4), letters, digits e.g., 22CH10059
        if re.match(r"^\d{2,4}[A-Z]{1,4}\d{2,6}$", t, re.IGNORECASE):
            return t.upper()
    # fallback: return first token or the full stem
    return tokens[0].upper() if tokens else base.upper()

import re

def ingest(zip_path: str, out_dir: str) -> List[Dict]:
    ensure_dir(out_dir)
    files = unzip_to_dir(zip_path, out_dir)
    docs=[]
    for f in files:
        text = extract_text_from_file(f)
        fn = os.path.basename(f)
        roll = filename_to_roll(fn)
        docs.append({
            "id": roll,                 # use roll number as id
            "roll_number": roll,
            "path": f,
            "raw_text": text,
            "filename": fn
        })
    return docs
