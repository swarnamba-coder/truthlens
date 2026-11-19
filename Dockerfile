FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg and system libraries for multimedia analysis
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY truthlens/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY truthlens/backend /app

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
