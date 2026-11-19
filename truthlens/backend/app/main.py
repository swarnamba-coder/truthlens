from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from app import config
from app.api.routes import router
from app.database.database import Base, engine, SessionLocal
from app.database.seeder import seed_demo_data

@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(engine)
    try:
        db = SessionLocal()
        seed_demo_data(db)
        db.close()
    except Exception as e:
        print(f"Demo data seeding notice: {e}")
    yield

app = FastAPI(title="TruthLens API", version="0.2.0", description="Forensic Intelligence Platform", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    return JSONResponse({"error": {"code": exc.status_code, "message": str(exc.detail)}}, status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    msg = "; ".join(f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in exc.errors())
    return JSONResponse({"error": {"code": 422, "message": msg or "Invalid request"}}, status_code=422)

app.include_router(router, prefix="/api/v1")

STATIC = Path(__file__).resolve().parent.parent / "static"

@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
