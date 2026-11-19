#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/truthlens/backend"

if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "[ERROR] Python 3 is not installed or not on PATH."
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
fi

source .venv/bin/activate
echo "Installing requirements..."
pip install -r requirements.txt

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
fi

echo "======================================================="
echo "TruthLens is running!"
echo "Access the full website at: http://localhost:8000"
echo "API documentation:          http://localhost:8000/docs"
echo "======================================================="
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
