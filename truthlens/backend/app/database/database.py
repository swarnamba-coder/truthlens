from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def iso(dt):
    return dt.isoformat() + "Z" if dt else None

class Investigation(Base):
    __tablename__ = "investigations"
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    case_ref = Column(String, default="")
    transcript = Column(Text, default="")
    status = Column(String, default="created")  # created|processing|completed|failed
    review_status = Column(String, default="not_started")
    is_demo = Column(Integer, default=0)  # 1 for pre-seeded demonstration data, 0 for real-time live cases
    error = Column(Text)
    created_at = Column(DateTime, default=utcnow)

class EvidenceFile(Base):
    __tablename__ = "evidence_files"
    id = Column(String, primary_key=True)
    investigation_id = Column(String, ForeignKey("investigations.id"), index=True)
    kind = Column(String)
    original_name = Column(String)
    stored_path = Column(String)
    size = Column(Integer)
    sha256 = Column(String)
    uploaded_at = Column(DateTime, default=utcnow)

class AgentExecution(Base):
    __tablename__ = "agent_executions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    investigation_id = Column(String, ForeignKey("investigations.id"), index=True)
    name = Column(String)
    state = Column(String, default="waiting")  # waiting|processing|completed|failed|unavailable
    reason = Column(Text)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    output = Column(JSON)  # per-file status + metadata

class Finding(Base):  # holds analysis AND correlation findings (modality distinguishes them)
    __tablename__ = "findings"
    id = Column(String, primary_key=True)
    investigation_id = Column(String, ForeignKey("investigations.id"), index=True)
    modality = Column(String)
    data = Column(JSON)

class Report(Base):
    __tablename__ = "reports"
    investigation_id = Column(String, ForeignKey("investigations.id"), primary_key=True)
    path = Column(String)
    created_at = Column(DateTime, default=utcnow)
