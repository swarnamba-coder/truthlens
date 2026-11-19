import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE))
UPLOADS, REPORTS, ARTIFACTS = DATA_DIR / "uploads", DATA_DIR / "reports", DATA_DIR / "artifacts"
for _p in (UPLOADS, REPORTS, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)

# Database selection: Supabase PostgreSQL (via connection string) or local SQLite fallback
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")
DATABASE_URL = SUPABASE_DB_URL or os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'truthlens.db'}")

# Storage selection: Supabase Storage or local filesystem fallback
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "truthlens-evidence")

# AI & Model APIs (stored in environment variables)
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_ENABLED = bool(HUGGINGFACE_API_KEY)

# Operational limits
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "200"))
MAX_VIDEO_SECONDS = float(os.getenv("MAX_VIDEO_SECONDS", "300"))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "600"))
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL_SECONDS", "2"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "60"))
CORRELATION_WINDOW = float(os.getenv("CORRELATION_WINDOW_SECONDS", "2"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080,http://localhost:3000").split(",")]
AGENTS = ["image", "video", "audio", "text", "correlation", "report"]

# Candidate Model & Repository Evaluation Registry (8 criteria inspection)
TOOL_REGISTRY = [
    {
        "id": "exiftool",
        "name": "ExifTool / PyExifTool",
        "category": "Image",
        "license": "GPLv1 / Artistic License",
        "pretrained_weights": "Not applicable (rule-based parser)",
        "python_compatibility": "Python 3.8+ via pyexiftool or subprocess wrapper",
        "hardware_requirements": "CPU only (< 50MB RAM)",
        "inference_practical_24h": True,
        "media_type": "Image (JPEG, PNG, TIFF, HEIC, WebP, etc.) & Video metadata",
        "version_source": "Phil Harvey ExifTool / pyexiftool 0.5.x",
        "integration_status": "Active (with transparent PIL ExifTags fallback if binary absent)",
        "notes": "Gold standard for EXIF, XMP, IPTC, and camera device fingerprinting."
    },
    {
        "id": "opencv",
        "name": "OpenCV (opencv-python-headless)",
        "category": "Image / Video",
        "license": "Apache 2.0",
        "pretrained_weights": "Not applicable (computer vision library)",
        "python_compatibility": "Python 3.8 - 3.12+",
        "hardware_requirements": "CPU optimized (multi-threaded SIMD)",
        "inference_practical_24h": True,
        "media_type": "Images and Video frames",
        "version_source": "OpenCV 4.x (opencv-python-headless on PyPI)",
        "integration_status": "Active",
        "notes": "Used for frame matrix operations, Laplacian edge variance, and color space transformations."
    },
    {
        "id": "trufor",
        "name": "TruFor (ECCV 2022)",
        "category": "Image",
        "license": "Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)",
        "pretrained_weights": "trufor.pth.tar (~450MB checkpoint via Grip-Unina)",
        "python_compatibility": "Python 3.8 - 3.10 (strict PyTorch 1.12/2.0)",
        "hardware_requirements": "CUDA GPU with >= 6GB VRAM recommended. CPU execution requires 15-30s per image.",
        "inference_practical_24h": True,
        "media_type": "Images (JPEG, PNG, TIFF)",
        "version_source": "github.com/grip-unina/TruFor",
        "integration_status": "Evaluated: GPU/Licensing requirement; fallback to ELA re-compression outlier analysis",
        "notes": "Transparent fallback active: ELA block outlier locator + EXIF integrity check to avoid large unvetted research dependencies."
    },
    {
        "id": "ffmpeg",
        "name": "FFmpeg / FFprobe",
        "category": "Video",
        "license": "LGPL 2.1+ / GPL",
        "pretrained_weights": "Not applicable (multimedia framework)",
        "python_compatibility": "System binary called via subprocess argument list",
        "hardware_requirements": "CPU multi-threaded",
        "inference_practical_24h": True,
        "media_type": "Video (MP4, MOV, MKV, WebM, AVI)",
        "version_source": "FFmpeg 6.x / 7.x",
        "integration_status": "Active",
        "notes": "Extracts nominal frame timestamps, audio streams, container metadata, and frame difference metrics."
    },
    {
        "id": "genconvit",
        "name": "GenConViT (WACV 2024)",
        "category": "Video",
        "license": "Research Only (Non-Commercial)",
        "pretrained_weights": "Vision Transformer weights (~1.2 GB)",
        "python_compatibility": "Python 3.9 - 3.11 with PyTorch + Timm",
        "hardware_requirements": "NVIDIA CUDA GPU with >= 8GB VRAM mandatory for temporal ViT",
        "inference_practical_24h": True,
        "media_type": "Video frame sequences",
        "version_source": "Generative-Convolutional-Vision-Transformer (WACV 2024)",
        "integration_status": "Evaluated: Requires CUDA GPU; transparent fallback to inter-frame visual anomaly detection",
        "notes": "Transparent fallback active: Frame-to-frame pixel delta outlier locator with nominal timestamps."
    },
    {
        "id": "whisper",
        "name": "Faster-Whisper (OpenAI / CTranslate2)",
        "category": "Audio",
        "license": "MIT",
        "pretrained_weights": "Hugging Face / Systran int8 quantized models (~75MB for tiny)",
        "python_compatibility": "Python 3.8 - 3.12",
        "hardware_requirements": "Fast CPU inference with int8 quantization (or GPU)",
        "inference_practical_24h": True,
        "media_type": "Audio (WAV, MP3, M4A, FLAC, OGG)",
        "version_source": "github.com/SYSTRAN/faster-whisper (model: tiny / base)",
        "integration_status": "Active",
        "notes": "Extracts word-level timestamps used for cross-modal text-audio alignment."
    },
    {
        "id": "librosa",
        "name": "Librosa",
        "category": "Audio",
        "license": "ISC License",
        "pretrained_weights": "Not applicable (audio DSP library)",
        "python_compatibility": "Python 3.8 - 3.12+",
        "hardware_requirements": "CPU only (< 150MB RAM)",
        "inference_practical_24h": True,
        "media_type": "Audio files",
        "version_source": "Librosa 0.10.x / 0.11.x via PyPI",
        "integration_status": "Active",
        "notes": "Generates STFT spectrogram images and spectral flatness anomaly indicators."
    },
    {
        "id": "aasist",
        "name": "AASIST (Audio Anti-Spoofing)",
        "category": "Audio",
        "license": "BSD 3-Clause",
        "pretrained_weights": "ASVspoof 2021 baseline checkpoint (~95MB PyTorch)",
        "python_compatibility": "Python 3.8 - 3.11 with PyTorch & TorchAudio",
        "hardware_requirements": "CPU or GPU (< 1GB VRAM)",
        "inference_practical_24h": True,
        "media_type": "Single-channel 16kHz audio",
        "version_source": "github.com/clovaai/aasist (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention)",
        "integration_status": "Evaluated: Baseline model; fallback to spectral flatness & harmonic discontinuity heuristics",
        "notes": "Transparent acoustic artifact analysis flags synthetic voice cues without claiming absolute verification."
    },
    {
        "id": "c2pa",
        "name": "C2PA Python SDK (Content Authenticity Initiative)",
        "category": "Provenance",
        "license": "MIT / Apache 2.0",
        "pretrained_weights": "Not applicable (cryptographic verification library)",
        "python_compatibility": "Python 3.9 - 3.12",
        "hardware_requirements": "CPU only (< 30MB RAM)",
        "inference_practical_24h": True,
        "media_type": "Images and Video containing JUMBF manifests",
        "version_source": "c2pa-python on PyPI / Content Authenticity Initiative",
        "integration_status": "Active (checks for embedded C2PA/CAI manifests and reports missing status with disclaimer)",
        "notes": "Verifies cryptographic Content Credentials. Reports manifest absence without falsely asserting manipulation."
    },
    {
        "id": "gemini",
        "name": "Gemini 2.0 Flash / Pro (Google AI)",
        "category": "AI",
        "license": "Commercial API (Google Cloud / Google AI Terms)",
        "pretrained_weights": "Hosted by Google Cloud",
        "python_compatibility": "REST API via httpx",
        "hardware_requirements": "Zero local hardware requirements (cloud hosted)",
        "inference_practical_24h": True,
        "media_type": "Structured finding text and cross-modal correlation summaries (files never sent)",
        "version_source": "Google Generative AI REST API (model: gemini-2.0-flash)",
        "integration_status": "Configurable (via GEMINI_API_KEY environment variable)",
        "notes": "Generates plain-language examiner summaries and synthesis. Privacy-guaranteed: raw media files are never transmitted."
    },
    {
        "id": "huggingface",
        "name": "Hugging Face Inference Providers",
        "category": "AI / Hosted Models",
        "license": "Commercial / Open-source hosted endpoints",
        "pretrained_weights": "Hosted by Hugging Face / Serverless Inference",
        "python_compatibility": "REST API via httpx",
        "hardware_requirements": "Cloud hosted",
        "inference_practical_24h": True,
        "media_type": "Text, audio, and vision classification models",
        "version_source": "api-inference.huggingface.co",
        "integration_status": "Configurable (via HUGGINGFACE_API_KEY)",
        "notes": "Allows optional offloading of large deepfake models to hosted serverless endpoints."
    },
    {
        "id": "supabase",
        "name": "Supabase PostgreSQL & Storage",
        "category": "Database & Storage",
        "license": "Apache 2.0 (Open Source / Hosted PaaS)",
        "pretrained_weights": "Not applicable (PostgreSQL + S3 Storage)",
        "python_compatibility": "Python SQLAlchemy (psycopg2 / asyncpg) + REST",
        "hardware_requirements": "Cloud hosted (Supabase.com)",
        "inference_practical_24h": True,
        "media_type": "Relational data & binary file blobs",
        "version_source": "PostgreSQL 15+ / Supabase Storage API",
        "integration_status": "Configurable (via SUPABASE_DB_URL & SUPABASE_KEY; local SQLite/file fallback active)",
        "notes": "Seamless switch from local SQLite to production Supabase PostgreSQL without code changes."
    },
    {
        "id": "reportlab",
        "name": "ReportLab PDF Engine",
        "category": "Reporting",
        "license": "BSD License",
        "pretrained_weights": "Not applicable (document generator)",
        "python_compatibility": "Python 3.8 - 3.12+",
        "hardware_requirements": "CPU only (< 80MB RAM)",
        "inference_practical_24h": True,
        "media_type": "PDF forensic dossiers",
        "version_source": "ReportLab 4.x / 5.x on PyPI",
        "integration_status": "Active",
        "notes": "Generates complete court-ready PDF forensic dossiers with cryptographic SHA-256 evidence inventories."
    }
]
