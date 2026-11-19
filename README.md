# TruthLens — Forensic Intelligence Platform

TruthLens is a digital forensic intelligence platform designed around evidence traceability, candidate tool evaluation, and explainability. It examines multimodal evidence (image, video, audio, and text) for indicators of manipulation and synthetic alteration, links timestamped observations across modalities, and compiles findings into court-ready ReportLab PDF dossiers.

---

## Model, Repository, API & Database Evaluation Matrix

Before integrating external repositories and machine learning models, TruthLens evaluates each candidate across 8 governance criteria:
1. **README Inspection**
2. **License Compliance**
3. **Pretrained-Weight Availability**
4. **Python Compatibility**
5. **Hardware Requirements**
6. **Inference Turnaround (< 24 Hours)**
7. **Intended Media Type Support**
8. **Version & Source Traceability**

If a model is incompatible, excessively heavy, or unavailable, **a transparent fallback is used rather than fabricating a detection result**.

| Modality / Domain | Candidate Project | License | Hardware Requirement | Pretrained Weights / Source | Turnaround (<24h) | Integration & Fallback Strategy |
|---|---|---|---|---|---|---|
| **IMAGE (Metadata)** | **ExifTool** / PyExifTool | GPLv1 / Artistic | CPU (< 50MB RAM) | Rule-based parser (Phil Harvey) | Immediate (< 1s) | **Active**: Native `exiftool` subprocess extraction with transparent PIL `ExifTags` fallback. |
| **IMAGE (Processing)** | **OpenCV** (`opencv-python-headless`) | Apache 2.0 | CPU (SIMD) | Computer vision library (v4.x) | Immediate (< 1s) | **Active**: Laplacian sharpness variance, color transforms, and spatial filtering. |
| **IMAGE (Forgery Localization)** | **TruFor** (ECCV 2022) | CC BY-NC 4.0 (Non-commercial) | CUDA GPU (>= 6GB VRAM) | ~450MB checkpoint (`trufor.pth.tar`) | Yes (GPU) / Slow (CPU) | **Evaluated**: Due to CC BY-NC 4.0 license and heavy GPU requirement, TruthLens executes a transparent **Error Level Analysis (ELA) block outlier fallback** with localized bounding box coordinates. |
| **VIDEO (Media Processing)** | **FFmpeg / FFprobe** | LGPL 2.1+ / GPL | CPU multi-threaded | Standard multimedia binary (v6+) | Immediate (< 5s) | **Active**: Keyframe extraction, container metadata, and nominal sampling. |
| **VIDEO (Deepfake Detection)** | **GenConViT** (WACV 2024) | Research Only | CUDA GPU (>= 8GB VRAM) | ~1.2GB Vision Transformer weights | Yes (GPU) / Impractical on CPU | **Evaluated**: Avoids unvetted research dependencies; uses transparent **frame-to-frame pixel delta discontinuity fallback** with nominal timestamps. |
| **AUDIO (Transcription)** | **Faster-Whisper** (OpenAI / SYSTRAN) | MIT | CPU int8 (or GPU) | Hugging Face int8 quantized (`tiny`/`base`) | Fast (< 10s) | **Active**: CTranslate2 int8 transcription with word-level timestamps. |
| **AUDIO (Feature Analysis)** | **Librosa** | ISC License | CPU (< 150MB RAM) | Signal processing library | Fast (< 2s) | **Active**: STFT spectrogram generation and spectral flatness computation. |
| **AUDIO (Anti-Spoofing)** | **AASIST** (ASVspoof 2021) | BSD 3-Clause | CPU or GPU (< 1GB VRAM) | ~95MB graph attention model | Fast | **Evaluated**: Baseline graph attention model; transparent fallback to **spectral flatness spike & high-frequency rolloff heuristics**. |
| **PROVENANCE** | **C2PA Python SDK** (CAI) | MIT / Apache 2.0 | CPU (< 30MB RAM) | Cryptographic verification parser | Immediate (< 1s) | **Active**: Inspects JUMBF/C2PA manifests. Reports absent manifests with explicit limitation: *absence does not prove fraud*. |
| **AI (Explanations)** | **Gemini API** (Google AI) | Commercial API | Cloud hosted | Google Cloud hosted (`gemini-2.0-flash`) | Fast (< 3s) | **Configurable**: Activated via `GEMINI_API_KEY`. Privacy guaranteed: finding text only is sent; raw files are never transmitted. |
| **AI (Hosted Inference)** | **Hugging Face Inference Providers** | Open-source / Commercial | Cloud hosted | Hosted serverless endpoints | Fast | **Configurable**: Configured via `HUGGINGFACE_API_KEY` for optional cloud deepfake models. |
| **DATABASE** | **Supabase PostgreSQL** | Apache 2.0 | Cloud hosted | PostgreSQL 15+ | Real-time | **Configurable**: Seamless switch to Supabase PostgreSQL via `DATABASE_URL` / `SUPABASE_DB_URL`. Defaults to local SQLite `truthlens.db`. |
| **FILE STORAGE** | **Supabase Storage** | Apache 2.0 | Cloud hosted | S3 / REST private buckets | Fast | **Configurable**: Uses `SUPABASE_URL` and `SUPABASE_KEY`; transparent fallback to local disk storage. |
| **REPORTING** | **ReportLab** | BSD License | CPU (< 80MB RAM) | Python document library (v4.x/5.x) | Immediate (< 2s) | **Active**: Generates court-ready PDF forensic dossiers with SHA-256 evidence verification. |

---

## 3 Pre-Seeded Demonstration Cases + Real-Time Analysis

TruthLens includes **3 pre-seeded, fully populated demonstration cases** alongside support for **real-time live investigations**:

1. **`TL-DEMO-01` — Press Briefing Audio/Video Manipulation (Video + Audio + Text)**:
   * Spliced press briefing video at timestamp `00:12`.
   * Video compression discontinuity (Frame 288) aligns with an audio spectral flatness drop at `00:12` and a difflib transcript omission of the phrase *"Our lawyers confirmed it"*.
   * Cross-modal correlation engine links all three findings into a single temporal event.
2. **`TL-DEMO-02` — Contract Agreement Scan Tampering (Image / Document)**:
   * Scanned purchase agreement. Error Level Analysis (ELA) identifies high-frequency re-compression in the signature block (`[x: 350, y: 250, w: 150, h: 80]`).
   * ExifTool detects `Software: Adobe Photoshop 24.1 (Windows)`. C2PA manifest check confirms no cryptographic Content Credentials.
3. **`TL-DEMO-03` — Synthetic Voice Phishing Voicemail (Audio + Text)**:
   * Urgent executive voicemail requesting an offshore wire transfer.
   * Librosa spectrogram and acoustic heuristics detect an unnatural high-frequency rolloff above 7.5 kHz characteristic of neural vocoder synthesis. Faster-Whisper provides transcription.
4. **Real-Time Live Analysis**:
   * Any new files uploaded via the **"New Investigation"** tab are processed dynamically in real-time by the backend orchestrator, hashed with SHA-256, and displayed with a green `REAL-TIME` badge.

---

## Provenance Tiers Explained

TruthLens categorizes every finding into one of four distinct provenance tiers:
* 🔍 **OBSERVED**: Direct cryptographic or file properties (EXIF tags, video fps, audio sample rate, SHA-256 digests).
* 📐 **HEURISTIC**: Algorithmic mathematical signals (ELA block outliers, spectral flatness spikes, frame-to-frame pixel differences).
* 🤖 **MODEL PREDICTION**: Machine learning model inferences (Faster-Whisper speech-to-text transcription).
* 💡 **AI EXPLANATION**: Natural language examiner guidance generated by Gemini 2.0.

> **Forensic Integrity Notice:** TruthLens never combines scores into a fake "manipulation probability". Each finding preserves its detector source, license, raw score, forensic limitations, and alternative benign explanations.

---

## Environment Variables & Credentials

All external credentials remain in backend environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///truthlens.db` | PostgreSQL / Supabase connection string. |
| `SUPABASE_DB_URL` | *(empty)* | Direct Supabase PostgreSQL URI. |
| `SUPABASE_URL` | *(empty)* | Supabase project API URL. |
| `SUPABASE_KEY` | *(empty)* | Supabase service role key for private storage. |
| `GEMINI_API_KEY` | *(empty)* | Google Generative AI API key for explanations. |
| `GEMINI_ENABLED` | `false` | Enable or disable Gemini AI synthesis. |
| `HUGGINGFACE_API_KEY` | *(empty)* | Optional Hugging Face Inference API token. |

---

## Quick Start

### Option 1: Docker (Recommended)
```cmd
docker compose up -d
```
Open **`http://localhost:8000`** in your browser.

### Option 2: Windows (Direct Local Run)
```cmd
.\run.bat
```
Open **`http://localhost:8000`**.
