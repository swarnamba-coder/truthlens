"""Unified evidence schema. Every finding carries explicit provenance; scores are raw detector output and are never combined into a single authenticity score."""
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

def new_finding(modality: str, file_id: Optional[str], provenance: str, title: str, description: str, *,
                detector: Optional[Dict[str, Any]] = None, score: Optional[float] = None,
                timestamp_seconds: Optional[float] = None, frame_number: Optional[int] = None,
                region: Optional[Dict[str, float]] = None, artifact: Optional[str] = None,
                limitations: Optional[List[str]] = None, alternative_explanations: Optional[List[str]] = None,
                related_finding_ids: Optional[List[str]] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12],
        "modality": modality,  # image | video | audio | text | correlation | provenance
        "evidence_file_id": file_id,
        "provenance": provenance,  # observed | heuristic | model_prediction | ai_explanation
        "title": title,
        "description": description,
        "detector": detector or {"name": "standard-inspector", "version": "1.0", "execution_mode": "observed_metadata"},
        "score": score,
        "timestamp_seconds": timestamp_seconds,
        "frame_number": frame_number,
        "region": region,  # {x, y, width, height}
        "artifact_path": artifact,
        "limitations": limitations or [],
        "alternative_explanations": alternative_explanations or [],
        "related_finding_ids": related_finding_ids or [],
        "tags": tags or [],
        "review_status": "not_reviewed"  # not_reviewed | confirmed | dismissed
    }

def result(status: str, reason: Optional[str] = None, findings: Optional[List[Dict[str, Any]]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "status": status,  # completed | failed | unavailable
        "reason": reason,
        "findings": findings or [],
        "metadata": metadata or {}
    }

class AgentOut(BaseModel):
    name: str
    state: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None

class StatusOut(BaseModel):
    investigation_id: str
    status: str
    progress_percent: int
    agents: List[AgentOut]

class ReviewIn(BaseModel):
    review_status: str  # not_reviewed | confirmed | dismissed

class CaseReviewIn(BaseModel):
    review_status: str  # not_started | in_progress | needs_review | reviewed
    examiner_notes: Optional[str] = None
