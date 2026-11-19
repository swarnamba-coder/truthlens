"""Lightweight orchestrator: runs the six agents in order, records real state, and never fakes a result."""
from pathlib import Path
from app.config import AGENTS, ARTIFACTS, CORRELATION_WINDOW
from app.database.database import SessionLocal, Investigation, EvidenceFile, AgentExecution, Finding, Report, utcnow
from app.engines import image_engine, video_engine, audio_engine, text_engine
from app.correlation.correlation_engine import correlate
from app.services import gemini_service, report_service
from app.schemas.findings import new_finding, result
from app.utils.logger import log

def _set(db, inv_id, name, **kw):
    ex = db.query(AgentExecution).filter_by(investigation_id=inv_id, name=name).one()
    for k, v in kw.items():
        setattr(ex, k, v)
    db.commit()

def _url(inv_id, name):
    return f"/api/v1/investigations/{inv_id}/artifacts/{name}" if name else None

def _run_agent(db, inv_id, name, jobs, ctx):
    """jobs: list of (label, call). Each call returns an engine result dict."""
    _set(db, inv_id, name, state="processing", started_at=utcnow(), reason=None, output=None)
    if not jobs:
        _set(db, inv_id, name, state="unavailable", reason=f"No {name} input submitted", finished_at=utcnow())
        return
    outs, states = {}, []
    for label, call in jobs:
        try:
            res = call()
        except Exception as e:
            log.exception("agent %s failed on %s", name, label)
            res = result("failed", f"{type(e).__name__}: {e}")
        for f in res["findings"]:
            f["artifact_path"] = _url(inv_id, f["artifact_path"])
            db.add(Finding(id=f["id"], investigation_id=inv_id, modality=f["modality"], data=f))
            ctx["findings"].append(f)
        if name == "audio":
            ctx["segments"].extend(res["metadata"].get("transcript", []))
        outs[label] = {"status": res["status"], "reason": res["reason"], "metadata": res["metadata"]}
        states.append(res["status"])
    db.commit()
    state = "completed" if "completed" in states else ("failed" if "failed" in states else "unavailable")
    reason = "; ".join(f"{k}: {v['reason']}" for k, v in outs.items() if v["reason"]) or None
    _set(db, inv_id, name, state=state, reason=reason, output=outs, finished_at=utcnow())

def run(inv_id: str):
    db = SessionLocal()
    try:
        inv = db.get(Investigation, inv_id)
        db.query(Finding).filter_by(investigation_id=inv_id).delete()
        for ex in db.query(AgentExecution).filter_by(investigation_id=inv_id):
            ex.state, ex.reason, ex.started_at, ex.finished_at, ex.output = "waiting", None, None, None, None
        inv.status, inv.error = "processing", None
        db.commit()
        files = db.query(EvidenceFile).filter_by(investigation_id=inv_id).all()
        art = ARTIFACTS / inv_id
        art.mkdir(parents=True, exist_ok=True)
        ctx = {"findings": [], "segments": []}
        engines = {"image": image_engine, "video": video_engine, "audio": audio_engine}
        for kind, eng in engines.items():
            jobs = [(f.original_name, (lambda f=f, eng=eng: eng.analyze(Path(f.stored_path), f.id, art)))
                    for f in files if f.kind == kind]
            _run_agent(db, inv_id, kind, jobs, ctx)
        text = inv.transcript or ""
        text_files = [f for f in files if f.kind == "text"]
        for f in text_files:
            text += "\n" + text_engine.extract_text(Path(f.stored_path))
        tid = text_files[0].id if text_files else None
        _run_agent(db, inv_id, "text", [("transcript comparison", lambda: text_engine.compare(text, ctx["segments"], tid))] if text.strip() else [], ctx)
        _run_agent(db, inv_id, "correlation", [("timestamp correlation", lambda: correlate(ctx["findings"], CORRELATION_WINDOW))], ctx)
        _report_agent(db, inv_id, ctx)
        inv.status = "completed" if db.query(AgentExecution).filter_by(investigation_id=inv_id, name="report").one().state == "completed" else "failed"
        db.commit()
    except Exception as e:
        log.exception("orchestrator crashed")
        db.rollback()
        inv = db.get(Investigation, inv_id)
        inv.status, inv.error = "failed", f"{type(e).__name__}: {e}"
        db.commit()
    finally:
        db.close()

def _report_agent(db, inv_id, ctx):
    _set(db, inv_id, "report", state="processing", started_at=utcnow())
    notes = []
    g = gemini_service.summarize(ctx["findings"]) if ctx["findings"] else {"status": "unavailable", "reason": "No findings to summarize", "text": None}
    if g["text"]:
        f = new_finding("correlation", None, "ai_explanation", "AI-generated summary", g["text"],
                        detector={"name": "gemini", "version": "api"}, limitations=["Generated text. Not a forensic result."])
        db.add(Finding(id=f["id"], investigation_id=inv_id, modality="correlation", data=f)); db.commit()
    else:
        notes.append(f"AI summary: {g['reason']}")
    try:
        path = report_service.build(db, inv_id)
        db.merge(Report(investigation_id=inv_id, path=str(path), created_at=utcnow())); db.commit()
        _set(db, inv_id, "report", state="completed", reason="; ".join(notes) or None, finished_at=utcnow())
    except Exception as e:
        log.exception("report failed")
        _set(db, inv_id, "report", state="failed", reason=f"{type(e).__name__}: {e}", finished_at=utcnow())
