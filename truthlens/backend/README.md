# TruthLens backend (FastAPI)

## Run
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # librosa / faster-whisper are optional; install FFmpeg separately
cp .env.example .env
uvicorn app.main:app --reload --port 8000     # docs at /docs
pytest
```
Check `GET /api/v1/health` to see which tools are really available (ffmpeg, librosa, faster_whisper, gemini).

## What actually runs
| Agent | Real work | If the tool is missing |
|---|---|---|
| image | Pillow decode check, EXIF, ELA block outliers (with pixel regions) | n/a (core deps) |
| video | ffprobe metadata, ffmpeg frames every N s, frame-difference outliers | state `unavailable` |
| audio | librosa features, spectrogram, spectral-flatness outliers, faster-whisper transcript | `unavailable` / transcription `unavailable` |
| text | word-level diff of supplied text vs audio transcript | `unavailable` (no input or no transcript) |
| correlation | groups findings from different modalities within a time window | no findings |
| report | ReportLab PDF from stored findings; optional Gemini summary | Gemini step `unavailable`, PDF still built |

No pretrained deepfake or synthetic-voice detector is integrated yet. Nothing in the API claims one is.
Scores are raw detector outputs and are never combined. Every finding carries `provenance`
(`observed | heuristic | model_prediction | ai_explanation`).

## Contract (matches the prototype)
`POST /investigations` multipart: `title`, optional `description`, `case_ref`, `transcript`, files in
`image`/`video`/`audio`/`text`. Returns 201 only after files are saved and hashed.
Also: `GET /investigations`, `GET /investigations/{id}`, `POST .../analyze` (202), `GET .../status`,
`GET .../results`, `GET .../report`, `GET .../evidence/{file_id}`, `GET .../artifacts/{path}`,
`PATCH .../findings/{finding_id}/review`, `GET /health`. Errors: `{"error":{"code","message"}}`.

## Known limits
- The Lovable frontend was not inspected; map its TypeScript interfaces to these shapes (Phase 1 of the engineering prompt).
- Video timestamps are nominal sampling times. Text engine reads `.txt`/`.md` only. Analysis runs sequentially in a background task.
- Gemini is off by default and receives finding text only, never files.
