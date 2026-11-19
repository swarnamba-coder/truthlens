import hashlib, shutil, uuid, importlib.util
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from app import config
from app.agents import orchestrator
from app.database.database import SessionLocal, Investigation, EvidenceFile, AgentExecution, Finding, Report, iso
from app.engines import video_engine
from app.schemas.findings import StatusOut, ReviewIn, CaseReviewIn
from app.utils.validation import ALLOWED, check_extension, safe_name

router = APIRouter()
DONE = ("completed", "failed", "unavailable")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _inv(db, inv_id):
    inv = db.get(Investigation, inv_id)
    if not inv:
        raise HTTPException(404, "Investigation not found")
    return inv

def _file_out(f):
    return {"id": f.id, "kind": f.kind, "name": f.original_name, "size": f.size, "sha256": f.sha256, "uploaded_at": iso(f.uploaded_at)}

def _summary(db, inv):
    files = db.query(EvidenceFile).filter_by(investigation_id=inv.id).all()
    return {"id": inv.id, "title": inv.title, "description": inv.description, "case_ref": inv.case_ref, "status": inv.status,
            "review_status": inv.review_status, "is_demo": bool(getattr(inv, "is_demo", 0)), "created_at": iso(inv.created_at), "error": inv.error,
            "media": sorted({f.kind for f in files}), "evidence": [_file_out(f) for f in files]}

@router.get("/health")
def health():
    return {
        "status": "ok",
        "services": {
            "ffmpeg": video_engine.tools_available(),
            "opencv": importlib.util.find_spec("cv2") is not None,
            "librosa": importlib.util.find_spec("librosa") is not None,
            "faster_whisper": importlib.util.find_spec("faster_whisper") is not None,
            "c2pa": importlib.util.find_spec("c2pa") is not None,
            "gemini_enabled": config.GEMINI_ENABLED and bool(config.GEMINI_API_KEY),
            "supabase_connected": bool(config.SUPABASE_DB_URL or (config.SUPABASE_URL and config.SUPABASE_KEY))
        }
    }

@router.get("/registry")
def model_registry():
    return {"registry": config.TOOL_REGISTRY}

@router.post("/investigations", status_code=201)
async def create(title: str = Form(...), description: str = Form(""), case_ref: str = Form(""), transcript: str = Form(""),
                 image: Optional[List[UploadFile]] = File(None), video: Optional[List[UploadFile]] = File(None),
                 audio: Optional[List[UploadFile]] = File(None), text: Optional[List[UploadFile]] = File(None),
                 db=Depends(get_db)):
    if not title.strip():
        raise HTTPException(422, "Investigation title is required")
    groups = {k: [u for u in (v or []) if u.filename] for k, v in (("image", image), ("video", video), ("audio", audio), ("text", text))}
    if not any(groups.values()):
        raise HTTPException(422, "Attach at least one evidence file")
    for kind, ups in groups.items():
        for u in ups:
            try:
                check_extension(kind, u.filename)
            except ValueError as e:
                raise HTTPException(422, str(e))
    inv_id = "TL-" + uuid.uuid4().hex[:6].upper()
    folder = config.UPLOADS / inv_id
    folder.mkdir(parents=True, exist_ok=True)
    rows, limit = [], config.MAX_FILE_MB * 1024 * 1024
    try:
        for kind, ups in groups.items():
            for u in ups:
                fid = uuid.uuid4().hex[:10]
                path = folder / f"{fid}_{safe_name(u.filename)}"
                h, size = hashlib.sha256(), 0
                with open(path, "wb") as out:
                    while chunk := await u.read(1 << 20):
                        size += len(chunk)
                        if size > limit:
                            raise HTTPException(413, f"'{u.filename}' exceeds {config.MAX_FILE_MB} MB")
                        h.update(chunk); out.write(chunk)
                rows.append(EvidenceFile(id=fid, investigation_id=inv_id, kind=kind, original_name=safe_name(u.filename),
                                         stored_path=str(path), size=size, sha256=h.hexdigest()))
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    inv = Investigation(id=inv_id, title=title.strip(), description=description, case_ref=case_ref, transcript=transcript)
    db.add(inv); db.add_all(rows)
    db.add_all(AgentExecution(investigation_id=inv_id, name=n) for n in config.AGENTS)
    db.commit()
    return _summary(db, inv)  # returned only after files are on disk and hashed

@router.get("/investigations")
def list_investigations(db=Depends(get_db)):
    return [_summary(db, i) for i in db.query(Investigation).order_by(Investigation.created_at.desc())]

@router.get("/investigations/{inv_id}")
def get_one(inv_id: str, db=Depends(get_db)):
    return _summary(db, _inv(db, inv_id))

@router.post("/investigations/{inv_id}/analyze", status_code=202)
def analyze(inv_id: str, bg: BackgroundTasks, db=Depends(get_db)):
    inv = _inv(db, inv_id)
    if inv.status == "processing":
        raise HTTPException(409, "Analysis already running")
    inv.status = "processing"
    db.commit()
    bg.add_task(orchestrator.run, inv_id)
    return {"id": inv_id, "status": "processing"}

@router.get("/investigations/{inv_id}/status", response_model=StatusOut)
def status(inv_id: str, db=Depends(get_db)):
    inv = _inv(db, inv_id)
    ex = db.query(AgentExecution).filter_by(investigation_id=inv_id).all()
    order = {n: i for i, n in enumerate(config.AGENTS)}
    ex.sort(key=lambda e: order.get(e.name, 99))
    done = sum(e.state in DONE for e in ex)
    return {"investigation_id": inv_id, "status": inv.status, "progress_percent": int(100 * done / max(len(ex), 1)),
            "agents": [{"name": e.name, "state": e.state, "started_at": iso(e.started_at), "finished_at": iso(e.finished_at),
                        "error": e.reason} for e in ex]}

@router.get("/investigations/{inv_id}/results")
def results(inv_id: str, db=Depends(get_db)):
    inv = _inv(db, inv_id)
    ex = {e.name: e for e in db.query(AgentExecution).filter_by(investigation_id=inv_id)}
    fs = [f.data for f in db.query(Finding).filter_by(investigation_id=inv_id)]
    out = {}
    for name in ("image", "video", "audio", "text", "correlation"):
        e = ex.get(name)
        out[name] = {"status": e.state if e else "waiting", "reason": e.reason if e else None,
                     "findings": [f for f in fs if f["modality"] == name], "metadata": (e.output if e else None) or {}}
    return {"investigation_id": inv_id, "status": inv.status, "results": out,
            "evidence": [_file_out(f) for f in db.query(EvidenceFile).filter_by(investigation_id=inv_id)]}

@router.patch("/investigations/{inv_id}/findings/{finding_id}/review")
def review(inv_id: str, finding_id: str, body: ReviewIn, db=Depends(get_db)):
    if body.review_status not in ("not_reviewed", "confirmed", "dismissed"):
        raise HTTPException(422, "review_status must be not_reviewed, confirmed or dismissed")
    f = db.query(Finding).filter_by(id=finding_id, investigation_id=inv_id).first()
    if not f:
        raise HTTPException(404, "Finding not found")
    f.data = {**f.data, "review_status": body.review_status}

    # Auto-update investigation overall review status based on finding progress
    inv = _inv(db, inv_id)
    all_findings = [row.data for row in db.query(Finding).filter_by(investigation_id=inv_id).all()]
    actionable = [x for x in all_findings if x.get("provenance") != "ai_explanation"]
    if actionable:
        reviewed_cnt = sum(1 for x in actionable if x.get("review_status") in ("confirmed", "dismissed"))
        if reviewed_cnt == len(actionable):
            inv.review_status = "reviewed"
        elif reviewed_cnt > 0:
            inv.review_status = "in_progress"
    db.commit()
    return f.data

@router.patch("/investigations/{inv_id}/review")
def review_case(inv_id: str, body: CaseReviewIn, db=Depends(get_db)):
    inv = _inv(db, inv_id)
    if body.review_status not in ("not_started", "in_progress", "needs_review", "reviewed"):
        raise HTTPException(422, "review_status must be not_started, in_progress, needs_review, or reviewed")
    inv.review_status = body.review_status
    if body.examiner_notes:
        note = f"\n[Examiner Sign-Off Note: {body.examiner_notes.strip()}]"
        inv.description = (inv.description or "") + note
    db.commit()
    return _summary(db, inv)

@router.get("/investigations/{inv_id}/report")
def report(inv_id: str, db=Depends(get_db)):
    _inv(db, inv_id)
    r = db.get(Report, inv_id)
    if not r:
        raise HTTPException(404, "Report not generated yet")
    return FileResponse(r.path, media_type="application/pdf", filename=f"{inv_id}-report.pdf")

@router.get("/investigations/{inv_id}/evidence/{file_id}")
def evidence(inv_id: str, file_id: str, db=Depends(get_db)):
    f = db.query(EvidenceFile).filter_by(id=file_id, investigation_id=inv_id).first()
    if not f:
        raise HTTPException(404, "Evidence file not found")
    return FileResponse(f.stored_path, filename=f.original_name)

@router.get("/investigations/{inv_id}/artifacts/{name:path}")
def artifact(inv_id: str, name: str, db=Depends(get_db)):
    _inv(db, inv_id)
    base = (config.ARTIFACTS / inv_id).resolve()
    p = (base / name).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(p)
