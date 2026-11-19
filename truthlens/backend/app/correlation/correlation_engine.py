"""Timestamp-window grouping across modalities. A match in time is NOT proof of a shared manipulation event."""
from app.schemas.findings import new_finding, result

ALT = {
 frozenset({"video", "audio"}): "A cut, re-encode or capture glitch can affect picture and sound at the same moment without any manipulation.",
 frozenset({"audio", "text"}): "Speech-to-text errors or noise in the recording can create a transcript mismatch at a flagged audio segment.",
 frozenset({"video", "text"}): "A scene change can coincide with a spoken passage that the supplied text words differently.",
}
GENERIC = "The observations may coincide by chance, or share a benign cause such as editing for length or compression."

def correlate(findings: list, window: float):
    timed = sorted((f for f in findings if f.get("timestamp_seconds") is not None and f["modality"] != "correlation"),
                   key=lambda f: f["timestamp_seconds"])
    clusters, cur = [], []
    for f in timed:
        if cur and f["timestamp_seconds"] - cur[-1]["timestamp_seconds"] > window:
            clusters.append(cur); cur = []
        cur.append(f)
    if cur:
        clusters.append(cur)
    out, linked = [], set()
    for c in clusters:
        mods = {f["modality"] for f in c}
        if len(mods) < 2:
            continue
        linked.update(f["id"] for f in c)
        alts = [ALT[frozenset(p)] for p in ALT if p <= mods] or [GENERIC]
        t = c[0]["timestamp_seconds"]
        item = new_finding("correlation", None, "heuristic", f"Related observations near {t:.1f}s",
            f"{len(c)} observations from {', '.join(sorted(mods))} fall within {window:g}s of each other. This is a timing match only.",
            timestamp_seconds=t, limitations=["Timing proximity does not show the observations share a cause."])
        item.update({"related_finding_ids": [f["id"] for f in c], "relation": "correlated", "alternative_explanations": alts})
        out.append(item)
    independent = [f["id"] for f in findings if f["id"] not in linked and f["modality"] != "correlation"]
    return result("completed", None, out, {"independent_finding_ids": independent, "window_seconds": window})
