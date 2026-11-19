FROM python:3.11-slim

WORKDIR /app

# Install system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libsndfile1 \
    exiftool \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in layers for better caching
# Layer 1: heavy ML libs (rarely change)
COPY truthlens/backend/requirements.txt .
RUN pip install --no-cache-dir \
    numpy>=1.26 \
    pillow>=10.0 \
    opencv-python-headless>=4.8 \
    librosa>=0.10 \
    soundfile>=0.12

# Layer 2: ML inference libs
RUN pip install --no-cache-dir \
    faster-whisper>=1.0 \
    c2pa-python>=0.5.0

# Layer 3: web + data libs
RUN pip install --no-cache-dir \
    fastapi>=0.110 \
    uvicorn[standard]>=0.29 \
    python-multipart>=0.0.9 \
    sqlalchemy>=2.0 \
    pydantic>=2.5 \
    reportlab>=4.0 \
    pypdf>=4.0 \
    httpx>=0.27 \
    python-dotenv>=1.0 \
    pytest>=8.0

COPY truthlens/backend /app

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
