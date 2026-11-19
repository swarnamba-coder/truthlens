"""Seeds TruthLens with 3 complete, realistic demonstration investigations and visual artifacts."""
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from app import config
from app.database.database import Investigation, EvidenceFile, AgentExecution, Finding, Report, utcnow
from app.schemas.findings import new_finding
from app.services import report_service

def _create_synthetic_spectrogram(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((256, 800, 3), dtype=np.uint8)
    for x in range(800):
        intensity = int(120 + 80 * np.sin(x * 0.05) * np.cos(x * 0.02))
        arr[80:200, x, 0] = np.clip(intensity, 0, 255)
        arr[80:200, x, 1] = np.clip(int(intensity * 0.7), 0, 255)
        arr[80:200, x, 2] = np.clip(int(intensity * 1.2), 0, 255)
    # Highlight anomaly near column 80 (timestamp 00:12)
    arr[60:220, 75:90, :] = [255, 100, 100]
    Image.fromarray(arr).save(path)

def _create_synthetic_ela(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((400, 600, 3), dtype=np.uint8)
    # Background low noise
    arr[:, :] = np.random.randint(10, 30, (400, 600, 3), dtype=np.uint8)
    # High ELA outlier at signature region [x: 350-500, y: 250-330]
    arr[250:330, 350:500] = np.random.randint(180, 255, (80, 150, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)

def _create_synthetic_frame(path: Path, frame_idx: int, timestamp: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (640, 360), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 620, 340], outline=(56, 189, 248), width=2)
    draw.text((40, 40), f"TruthLens Demo Video - Frame {frame_idx}", fill=(241, 245, 249))
    draw.text((40, 70), f"Nominal Timestamp: {timestamp:.2f}s", fill=(148, 163, 184))
    if 11.0 <= timestamp <= 13.0:
        draw.rectangle([200, 100, 440, 260], outline=(251, 191, 36), width=3)
        draw.text((210, 110), "FLAGGED DISCONTINUITY", fill=(251, 191, 36))
    img.save(path)

def seed_demo_data(db):
    """Ensures 3 rich demo investigations exist with complete artifacts and reports."""
    demos = [
        {
            "id": "TL-DEMO-01",
            "title": "[DEMO] Press Briefing Audio/Video Manipulation",
            "description": "Demonstration case examining a 2-minute press conference clip. Video compression analysis detects an abrupt cut at 00:12 (Frame 288), acoustic feature analysis detects a spectral gap at 00:12, and difflib comparison flags a missing sentence in the transcript.",
            "case_ref": "CASE-2026-PB01",
            "transcript": "We never agreed to that number. The meeting ended shortly after. Please confirm by Friday.",
            "files": [
                {"id": "ev-v01", "kind": "video", "name": "press_briefing_clip.mp4", "size": 14200000, "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
                {"id": "ev-a01", "kind": "audio", "name": "briefing_audio.wav", "size": 4100000, "sha256": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"},
                {"id": "ev-t01", "kind": "text", "name": "official_press_release.txt", "size": 1840, "sha256": "5946808753239a585721869d084776c9ad64b73b5f0d36e8fa17f18579cf6619"}
            ],
            "findings": [
                new_finding("video", "ev-v01", "observed", "Video Stream Parameters",
                            "1280x720 at 24.0 fps, 120.0s duration; 60 keyframes sampled every 2.0s.",
                            detector={"name": "FFmpeg/FFprobe", "version": "6.1", "source": "FFmpeg project", "license": "LGPL 2.1+", "execution_mode": "observed_metadata"},
                            limitations=["Nominal sampling interval of 2.0s may miss short sub-second splicing events."]),
                new_finding("video", "ev-v01", "heuristic", "Abrupt Compression Discontinuity at 00:12",
                            "Frame 288 shows anomalous inter-frame pixel difference (delta 28.45) compared to neighbouring frames.",
                            detector={"name": "GenConViT-Evaluated (Frame-Diff Fallback)", "version": "1.0", "source": "WACV 2024 / TruthLens", "license": "Research Fallback", "execution_mode": "transparent_fallback"},
                            score=28.45, timestamp_seconds=12.0, frame_number=288, artifact="TL-DEMO-01_frames/f_0006.jpg",
                            limitations=["Scene cuts, lighting shifts, or camera angle changes can produce the exact same metric.", "Does not prove intentional tampering by itself."],
                            alternative_explanations=["Camera switch in broadcast studio.", "Re-compression artifact during web streaming encode."]),
                new_finding("audio", "ev-a01", "observed", "Audio Stream Parameters",
                            "44.1 kHz, 16-bit stereo PCM; 120.0s total duration.",
                            detector={"name": "Librosa DSP", "version": "0.10.2", "source": "PyPI", "license": "ISC", "execution_mode": "observed_metadata"},
                            artifact="TL-DEMO-01_spectrogram.png"),
                new_finding("audio", "ev-a01", "heuristic", "Spectral Flatness Gap near 00:12",
                            "Sudden drop in high-frequency spectral flatness between 11.8s and 12.3s relative to surrounding speech.",
                            detector={"name": "AASIST-Evaluated (Spectral Flatness Fallback)", "version": "1.0", "source": "ASVspoof baseline / TruthLens", "license": "BSD-3", "execution_mode": "transparent_fallback"},
                            score=0.842, timestamp_seconds=12.0, artifact="TL-DEMO-01_spectrogram.png",
                            limitations=["Microphone breathing noise or sudden acoustic silence triggers similar flatness shifts."]),
                new_finding("text", "ev-t01", "observed", "Supplied Phrase Omitted in Audio Transcript",
                            "Supplied text contains: 'Our lawyers confirmed it', but this phrase is absent from the transcribed speech around 00:12.",
                            detector={"name": "Difflib-SequenceMatcher", "version": "Python 3 stdlib", "source": "Python", "license": "PSF", "execution_mode": "observed_metadata"},
                            timestamp_seconds=12.0,
                            limitations=["Speech-to-text recognition errors or spontaneous speaker omission can create identical discrepancies."]),
                new_finding("correlation", None, "heuristic", "Cross-Modal Correlation near 00:12",
                            "3 observations from Video, Audio, and Text coincide within 0.5s at timestamp 00:12. This temporal cluster indicates a concurrent event across modalities.",
                            timestamp_seconds=12.0,
                            related_finding_ids=[],
                            alternative_explanations=["A deliberate cut editing out a clarifying remark.", "A benign broadcast edit made to fit television time constraints."]),
                new_finding("correlation", None, "ai_explanation", "AI Synthesis & Examiner Guidance",
                            "Findings across video, audio, and text converge at 00:12. The video discontinuity aligns with an audio spectral gap and a transcript omission. While consistent with an edit, examiners should inspect raw camera uncompressed footage before concluding manipulation.",
                            detector={"name": "Gemini 2.0 Flash", "version": "2.0-flash", "source": "Google Cloud", "license": "Commercial API", "execution_mode": "api_provider"},
                            limitations=["Generated summary for human examiner guidance. Not a standalone forensic proof."])
            ]
        },
        {
            "id": "TL-DEMO-02",
            "title": "[DEMO] Contract Agreement Scan Tampering",
            "description": "Image forensics demonstration case. A scanned PDF/JPEG purchase contract agreement. Error Level Analysis (ELA) isolates a high-frequency compression anomaly around the signature box, and EXIF inspection reveals an Adobe Photoshop editing signature.",
            "case_ref": "CASE-2026-CT02",
            "transcript": "",
            "files": [
                {"id": "ev-img01", "kind": "image", "name": "signed_contract_scan.jpg", "size": 2850000, "sha256": "785b98bf6147494f6f890c01a2e41f0a15c7e3f8902123547841584758963214"}
            ],
            "findings": [
                new_finding("image", "ev-img01", "observed", "Image Resolution & EXIF Metadata",
                            "2480x3508 px JPEG image, 14 EXIF metadata tags preserved.",
                            detector={"name": "ExifTool / PyExifTool", "version": "12.70", "source": "Phil Harvey", "license": "GPLv1/Artistic", "execution_mode": "observed_metadata"},
                            limitations=["EXIF metadata can be easily stripped or edited by anyone; presence or absence is not conclusive."]),
                new_finding("image", "ev-img01", "observed", "Editing Software Tag Present",
                            "EXIF Software tag records 'Adobe Photoshop 24.1 (Windows)' with modification timestamp 2026-11-15T09:22:14Z.",
                            detector={"name": "ExifTool / PyExifTool", "version": "12.70", "source": "Phil Harvey", "license": "GPLv1/Artistic", "execution_mode": "observed_metadata"},
                            limitations=["Scanning software often uses Photoshop plugins for basic auto-deskewing and contrast enhancement."]),
                new_finding("image", "ev-img01", "heuristic", "Localized ELA Compression Outlier in Signature Region",
                            "Error Level Analysis demonstrates that pixels in block [x: 350, y: 250, w: 150, h: 80] re-compress with significantly higher residual error than the surrounding document.",
                            detector={"name": "TruFor-Evaluated (ELA Block Fallback)", "version": "1.0", "source": "ECCV 2022 / TruthLens", "license": "Research Fallback", "execution_mode": "transparent_fallback"},
                            score=3.42, region={"x": 350, "y": 250, "width": 150, "height": 80}, artifact="TL-DEMO-02_ela.png",
                            limitations=["High-contrast edges and pure black signatures naturally generate elevated ELA residuals.", "A human examiner must perform micro-level edge inspection."]),
                new_finding("provenance", "ev-img01", "observed", "C2PA Content Credentials Manifest Check",
                            "No C2PA/CAI cryptographic provenance manifest is embedded in this JPEG container.",
                            detector={"name": "C2PA Python SDK", "version": "0.5.0", "source": "Content Authenticity Initiative", "license": "MIT/Apache 2.0", "execution_mode": "observed_metadata"},
                            limitations=["Most office scanners and phones do not yet implement C2PA signing. Absence of C2PA manifest does NOT indicate forgery."])
            ]
        },
        {
            "id": "TL-DEMO-03",
            "title": "[DEMO] Synthetic Voice Phishing Voicemail",
            "description": "Audio anti-spoofing demonstration case. An urgent voicemail purporting to be the CFO requesting an immediate offshore wire transfer. Acoustic feature analysis reveals unnatural spectral flatness and robotic high-frequency rolloff.",
            "case_ref": "CASE-2026-VM03",
            "transcript": "Urgent message for payroll. Please authorize the wire transfer to the revised overseas routing code today.",
            "files": [
                {"id": "ev-aud01", "kind": "audio", "name": "urgent_cfo_voicemail.wav", "size": 1950000, "sha256": "9f83c2718e22d9921b0ff42b947a16499318b8fe1b3848b64b19e917d2a58b8f"},
                {"id": "ev-txt01", "kind": "text", "name": "verified_executive_communication.txt", "size": 850, "sha256": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"}
            ],
            "findings": [
                new_finding("audio", "ev-aud01", "observed", "Audio Codec and Sampling Parameters",
                            "16000 Hz mono WAV audio, 38.4s duration.",
                            detector={"name": "Librosa", "version": "0.10.2", "source": "PyPI", "license": "ISC", "execution_mode": "observed_metadata"},
                            artifact="TL-DEMO-03_spectrogram.png"),
                new_finding("audio", "ev-aud01", "heuristic", "Synthetic Voice Acoustic Anomaly (High-Frequency Rolloff)",
                            "AASIST acoustic feature evaluation detects a sharp, unnatural spectral cliff at 7.5 kHz with robotic formant periodicity consistent with neural vocoder synthesis.",
                            detector={"name": "AASIST-Evaluated (Acoustic Fallback)", "version": "1.0", "source": "ASVspoof 2021 / TruthLens", "license": "BSD-3", "execution_mode": "transparent_fallback"},
                            score=0.912, artifact="TL-DEMO-03_spectrogram.png",
                            limitations=["Aggressive cellular telephony codecs (e.g. AMR-NB) can also sharply bandlimit frequencies above 7 kHz."]),
                new_finding("audio", "ev-aud01", "model_prediction", "Faster-Whisper Speech Transcription",
                            "Transcribed speech content: 'Urgent message for payroll. Please authorize the wire transfer to the revised overseas routing code today.' Language detected: English (confidence 0.99).",
                            detector={"name": "Faster-Whisper", "version": "1.0 (CTranslate2 int8)", "source": "OpenAI / SYSTRAN", "license": "MIT", "execution_mode": "native_model"}),
                new_finding("correlation", None, "ai_explanation", "AI Phishing Synthesis & Advisory",
                            "The audio exhibits acoustic hallmarks of text-to-speech vocoding combined with urgency cues characteristic of business email compromise (BEC). Recommended action: verify caller identity via out-of-band secondary telephone protocol.",
                            detector={"name": "Gemini 2.0 Flash", "version": "2.0-flash", "source": "Google Cloud", "license": "Commercial API", "execution_mode": "api_provider"},
                            limitations=["AI heuristic explanation. Verification with the named executive is mandatory before law enforcement escalation."])
            ]
        }
    ]

    for d in demos:
        existing = db.get(Investigation, d["id"])
        if existing:
            continue
        inv = Investigation(
            id=d["id"],
            title=d["title"],
            description=d["description"],
            case_ref=d["case_ref"],
            transcript=d["transcript"],
            status="completed",
            review_status="needs_review",
            is_demo=1,
            created_at=utcnow()
        )
        db.add(inv)

        # Evidence files
        files_to_add = []
        for f in d["files"]:
            stored = config.UPLOADS / d["id"] / f["name"]
            stored.parent.mkdir(parents=True, exist_ok=True)
            if not stored.exists():
                stored.write_text(f"Demo file placeholder: {f['name']}\nSHA256: {f['sha256']}")
            files_to_add.append(EvidenceFile(
                id=f["id"],
                investigation_id=d["id"],
                kind=f["kind"],
                original_name=f["name"],
                stored_path=str(stored),
                size=f["size"],
                sha256=f["sha256"],
                uploaded_at=utcnow()
            ))
        db.add_all(files_to_add)

        # Agent executions
        for agent_name in config.AGENTS:
            db.add(AgentExecution(
                investigation_id=d["id"],
                name=agent_name,
                state="completed",
                started_at=utcnow(),
                finished_at=utcnow(),
                output={"demo": True, "status": "completed"}
            ))

        # Generate visual demo artifacts
        art_dir = config.ARTIFACTS / d["id"]
        art_dir.mkdir(parents=True, exist_ok=True)
        if d["id"] == "TL-DEMO-01":
            _create_synthetic_spectrogram(art_dir / "TL-DEMO-01_spectrogram.png")
            for i in range(1, 10):
                t = i * 2.0
                _create_synthetic_frame(art_dir / f"TL-DEMO-01_frames/f_{i:04d}.jpg", i * 48, t)
        elif d["id"] == "TL-DEMO-02":
            _create_synthetic_ela(art_dir / "TL-DEMO-02_ela.png")
        elif d["id"] == "TL-DEMO-03":
            _create_synthetic_spectrogram(art_dir / "TL-DEMO-03_spectrogram.png")

        # Findings
        for finding in d["findings"]:
            # Ensure artifact path has correct URL prefix if present
            if finding.get("artifact_path") and not finding["artifact_path"].startswith("/"):
                finding["artifact_path"] = f"/api/v1/investigations/{d['id']}/artifacts/{finding['artifact_path']}"
            db.add(Finding(
                id=finding["id"],
                investigation_id=d["id"],
                modality=finding["modality"],
                data=finding
            ))

        db.commit()

        # Build official PDF report for each demo
        try:
            report_path = report_service.build(db, d["id"])
            db.merge(Report(investigation_id=d["id"], path=str(report_path), created_at=utcnow()))
            db.commit()
        except Exception as e:
            print(f"Warning: demo report build for {d['id']} failed: {e}")
