"""Text analysis: compare a supplied transcript with the audio transcript (word-level difflib)."""
import re, difflib, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
from app.schemas.findings import new_finding, result

DET = {"name": "difflib-word-diff", "version": "1"}
LIM = ["Speech-to-text errors, paraphrasing and incomplete supplied text all create differences.",
       "A difference is not evidence of manipulation."]

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    elif ext == ".docx":
        try:
            with zipfile.ZipFile(path) as z:
                xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                texts = [node.text for node in tree.iter() if node.text]
                return "\n".join(texts)
        except Exception:
            return ""
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    return ""

def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())

def compare(supplied: str, segments: list, file_id=None):
    if not supplied.strip():
        return result("unavailable", "No supplied transcript or text file")
    if not segments:
        return result("unavailable", "No audio transcript available to compare against")
    a, b, bt = _words(supplied), [], []
    for s in segments:
        for w in _words(s["text"]):
            b.append(w); bt.append(s["start"])
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ratio = round(sm.ratio(), 3)
    findings = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or max(i2 - i1, j2 - j1) < 3 or len(findings) >= 20:
            continue
        t = bt[min(j1, len(bt) - 1)] if bt else None
        sup, aud = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        kind = {"delete": "Supplied phrase not found in audio transcript", "insert": "Audio phrase not in supplied text",
                "replace": "Supplied and audio wording differ"}[tag]
        findings.append(new_finding("text", file_id, "observed", kind,
            f"Supplied: \"{sup or '-'}\". Audio: \"{aud or '-'}\".", detector=DET, timestamp_seconds=t, limitations=LIM))
    return result("completed", None, findings, {"similarity_ratio": ratio, "supplied_words": len(a), "audio_words": len(b)})
