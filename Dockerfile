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
    soundfile>=0.12 \
    scipy>=1.11

# Layer 2: PyTorch CPU (AI image detection)
RUN pip install --no-cache-dir \
    torch>=2.1 torchvision>=0.16 \
    --index-url https://download.pytorch.org/whl/cpu

# Layer 3: ML inference + AI detection libs
RUN pip install --no-cache-dir \
    faster-whisper>=1.0 \
    transformers>=4.40 \
    timm>=0.9 \
    safetensors>=0.4 \
    huggingface-hub>=0.20 \
    tokenizers>=0.19 \
    regex>=2023.0 \
    scikit-learn>=1.3

# Layer 4: web + data libs
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

# Pre-download the ViT AI-detection model at build time (avoids cold-start delay)
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='Organika/sdxl-detector', device=-1)" || echo "Model pre-download skipped"

COPY truthlens/backend /app

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
