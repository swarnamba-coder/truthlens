# TruthLens

Backend (FastAPI) plus the TruthLens interface, served together on one port.

## Start
- Windows: double-click `run.bat` (or run it in the VS Code terminal)
- Mac/Linux: `./run.sh`

Then open **http://localhost:8000**. API docs: http://localhost:8000/docs

Requirements: Python 3.10+, and FFmpeg on your PATH for video analysis.
If `librosa` or `faster-whisper` fail to install, remove those two lines from `backend/requirements.txt`;
the audio agent will then say "unavailable" instead of failing.

## Using it
1. Open Settings and press Test connection (top bar should say "Backend connected").
2. New investigation: add a title and files, press Start analysis. Progress shows real agent states.
3. Open results from Investigations: Evidence tabs, Correlation, then Download PDF on Reports.
4. Demo mode (Settings) shows labelled synthetic data and never calls the backend for results.

## Tests
`cd backend && pytest`

See `backend/README.md` for what each agent really does, and what is not implemented
(no pretrained deepfake or synthetic-voice detector yet).
