"""Optional Gemini explanations. Receives finding TEXT only (never files). Disabled unless GEMINI_ENABLED=true."""
import httpx
from app import config

PROMPT = ("You are assisting a digital forensics examiner. Using ONLY the findings below, write a short plain-language summary. "
          "Do not state that any media is fake or authentic, do not invent facts, and mention that findings need human review.\n\n")

def summarize(findings: list):
    if not config.GEMINI_ENABLED:
        return {"status": "unavailable", "reason": "Gemini is disabled (GEMINI_ENABLED=false)", "text": None}
    if not config.GEMINI_API_KEY:
        return {"status": "unavailable", "reason": "GEMINI_API_KEY is not set", "text": None}
    lines = "\n".join(f"- [{f['modality']}/{f['provenance']}] {f['title']}: {f['description']}" for f in findings[:40])
    try:
        r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent",
                       headers={"x-goog-api-key": config.GEMINI_API_KEY},
                       json={"contents": [{"parts": [{"text": PROMPT + lines}]}]}, timeout=25)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"status": "completed", "reason": None, "text": text}
    except Exception as e:
        return {"status": "failed", "reason": f"Gemini request failed: {type(e).__name__}", "text": None}
