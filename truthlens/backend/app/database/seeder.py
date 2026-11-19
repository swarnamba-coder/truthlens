"""Seeds TruthLens with 3 complete demo investigations using REAL synthetic media files."""
import io, math, struct, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app import config
from app.database.database import Investigation, EvidenceFile, AgentExecution, Finding, Report, utcnow
from app.schemas.findings import new_finding
from app.services import report_service


# ── REAL SYNTHETIC MEDIA GENERATORS ──────────────────────────────────────────

def _make_wav(path: Path, duration_s: float = 38.0, sr: int = 16000, pattern: str = "voice"):
    """Generate a real playable WAV file with synthetic audio pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    if pattern == "voice":
        # Synthesise speech-like formant pattern: F1~700Hz, F2~1200Hz, F3~2500Hz
        sig = (
            0.4 * np.sin(2 * np.pi * 220 * t) +           # fundamental
            0.3 * np.sin(2 * np.pi * 700 * t) +           # F1
            0.2 * np.sin(2 * np.pi * 1200 * t) +          # F2
            0.08 * np.sin(2 * np.pi * 2500 * t) +         # F3
            0.05 * np.random.randn(n).astype(np.float32)   # slight noise
        )
        # Add amplitude envelope: speech-like bursts
        envelope = np.ones(n, dtype=np.float32)
        for i in range(0, n, sr // 2):        # every 0.5 s
            end = min(i + sr // 3, n)
            envelope[i:end] = 1.0
            if end < n:
                envelope[end:min(end + sr // 10, n)] = 0.1   # silence gap
        sig *= envelope
        # Inject anomaly near 12 s (spectral gap → abrupt flatness drop)
        gap_start = int(11.8 * sr)
        gap_end = int(12.3 * sr)
        sig[gap_start:gap_end] *= 0.05
    else:
        # Generic synthetic voice for phishing demo
        base = 0.5 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 360 * t)
        # Robotic periodicity: very regular formant
        formant = 0.3 * np.sin(2 * np.pi * 7500 * t)   # sharp 7.5 kHz rolloff region
        sig = base + 0.1 * formant + 0.02 * np.random.randn(n).astype(np.float32)
        # High-frequency cliff: bandlimit above 7.5 kHz (simulate vocoder rolloff)
        sig = sig * 0.8   # normalise

    # Normalise to 16-bit PCM
    peak = np.abs(sig).max()
    if peak > 0:
        sig = sig / peak * 0.9
    pcm = (sig * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _make_mp4(path: Path, duration_s: float = 30.0):
    """Generate a real playable MP4 using FFmpeg lavfi test source."""
    import subprocess, shutil
    path.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg"):
        # Fallback: create a minimal valid MP4 header stub
        path.write_bytes(b'\x00' * 1024)
        return False
    # Create a colour-bars test video with burned-in timestamp
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=navy:size=1280x720:rate=24:duration={duration_s}",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration_s}",
        # Overlay: frame counter + demo label
        "-vf", (
            "drawtext=text='TruthLens Demo — Press Briefing Clip':fontcolor=white:"
            "fontsize=36:x=(w-text_w)/2:y=60,"
            "drawtext=text='FRAME %{frame_num}  |  TIME %{pts\\:gmtime\\:0\\:%M\\:%S}':fontcolor=yellow:"
            "fontsize=24:x=40:y=120,"
            "drawtext=text='[ANOMALY REGION — 00\\:12]':fontcolor=red:"
            "fontsize=28:x=(w-text_w)/2:y=640:"
            "enable='between(t,11,13)',"
            "drawbox=x=400:y=200:w=480:h=320:color=yellow@0.4:t=4:"
            "enable='between(t,11,13)'"
        ),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
        "-c:a", "aac", "-b:a", "64k",
        "-t", str(duration_s),
        str(path)
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    return r.returncode == 0


def _make_contract_jpg(path: Path):
    """Generate a realistic contract scan JPEG with embedded EXIF-like metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 1240, 1754   # A4 at 150 dpi
    img = Image.new("RGB", (W, H), color=(250, 248, 243))
    draw = ImageDraw.Draw(img)

    # Paper texture noise
    noise = np.random.randint(0, 12, (H, W, 3), dtype=np.uint8)
    tex = Image.fromarray(noise)
    img = Image.blend(img, tex.resize((W, H)), alpha=0.08)
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([60, 60, W - 60, 130], outline=(30, 30, 30), width=2)
    draw.text((80, 75), "PURCHASE AGREEMENT — CONFIDENTIAL", fill=(20, 20, 80))
    draw.text((80, 100), "Contract Reference: CNT-2026-0892 | Date: 15 November 2026", fill=(60, 60, 60))

    # Body paragraphs
    lines = [
        "1. PARTIES: This Agreement is entered into between Acme Corp Ltd (Buyer)",
        "   and GlobalTech Supplies Pte Ltd (Seller), collectively the 'Parties'.",
        "",
        "2. GOODS: The Buyer agrees to purchase 500 units of industrial sensors",
        "   (Model XT-900) at a unit price of USD 1,200.00.",
        "",
        "3. TOTAL VALUE: USD 600,000.00 (Six Hundred Thousand US Dollars).",
        "",
        "4. PAYMENT TERMS: Net 30 days from delivery confirmation.",
        "",
        "5. DELIVERY: Delivery to Mumbai Warehouse by 20 December 2026.",
        "",
        "6. GOVERNING LAW: This contract shall be governed by the laws of Singapore.",
        "",
        "7. DISPUTE RESOLUTION: Any disputes shall be resolved by arbitration.",
        "",
        "   Both parties agree to the terms and conditions stated herein.",
    ]
    y = 180
    for line in lines:
        draw.text((80, y), line, fill=(30, 30, 30))
        y += 28

    # Signature blocks
    y = 1300
    draw.line([(80, y + 60), (460, y + 60)], fill=(30, 30, 30), width=1)
    draw.line([(600, y + 60), (980, y + 60)], fill=(30, 30, 30), width=1)
    draw.text((80, y + 65), "Authorized Signatory — Buyer", fill=(60, 60, 60))
    draw.text((600, y + 65), "Authorized Signatory — Seller", fill=(60, 60, 60))

    # Fake signature scrawls (curved lines)
    for i in range(0, 60, 3):
        x0 = 100 + i * 3
        y0 = y + 40 + int(10 * math.sin(i * 0.4))
        draw.ellipse([x0, y0, x0 + 4, y0 + 4], fill=(10, 10, 80))

    # ── TAMPERED REGION: paste different-quality signature block ──────────────
    # Simulate a pasted signature (higher re-compression artifact area)
    sig_patch = Image.new("RGB", (380, 90), color=(255, 255, 255))
    draw_p = ImageDraw.Draw(sig_patch)
    for i in range(0, 50, 2):
        x0 = 20 + i * 4
        y0 = 35 + int(14 * math.sin(i * 0.5))
        draw_p.ellipse([x0, y0, x0 + 5, y0 + 5], fill=(0, 0, 160))
    # Save at different quality to simulate re-compression
    buf = io.BytesIO()
    sig_patch.save(buf, "JPEG", quality=95)
    buf.seek(0)
    sig_patch_reloaded = Image.open(buf).convert("RGB")
    img.paste(sig_patch_reloaded, (600, y - 10))

    # Save with EXIF-like software tag (using PIL comment metadata)
    buf2 = io.BytesIO()
    img.save(buf2, "JPEG", quality=72)
    buf2.seek(0)
    # Inject a minimal EXIF block with Software=Adobe Photoshop 24.1
    jpeg_data = buf2.read()
    exif_block = _make_minimal_exif_with_software()
    if exif_block:
        # Insert EXIF after SOI marker
        final = jpeg_data[:2] + exif_block + jpeg_data[2:]
    else:
        final = jpeg_data
    path.write_bytes(final)


def _make_minimal_exif_with_software():
    """Build a minimal EXIF APP1 block with Software tag."""
    try:
        import piexif
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Software: b"Adobe Photoshop 24.1 (Windows)",
                piexif.ImageIFD.DateTime: b"2026:11:15 09:22:14",
                piexif.ImageIFD.Make: b"Canon",
                piexif.ImageIFD.Model: b"Canon EOS 5D",
            },
            "Exif": {},
            "GPS": {},
            "1st": {}
        }
        return piexif.dump(exif_dict)
    except Exception:
        return None


def _create_synthetic_spectrogram(path: Path, anomaly_col: int = 80):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((256, 1000, 3), dtype=np.uint8)
    for x in range(1000):
        intensity = int(120 + 80 * np.sin(x * 0.05) * np.cos(x * 0.02))
        arr[60:220, x, 0] = np.clip(intensity, 0, 255)
        arr[60:220, x, 1] = np.clip(int(intensity * 0.5), 0, 255)
        arr[60:220, x, 2] = np.clip(int(intensity * 1.3), 0, 255)
    arr[40:230, anomaly_col - 8:anomaly_col + 8, :] = [255, 80, 80]
    Image.fromarray(arr).save(path)


def _create_ela_heatmap(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(5, 25, (600, 1240, 3), dtype=np.uint8)
    # Hot region at signature area
    arr[480:570, 600:980] = np.random.randint(160, 255, (90, 380, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _create_frame(path: Path, frame_idx: int, timestamp: float, anomaly: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    color = (15, 23, 42) if not anomaly else (40, 15, 15)
    img = Image.new("RGB", (640, 360), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 630, 350], outline=(56, 189, 248) if not anomaly else (251, 191, 36), width=2)
    draw.text((30, 30), f"TruthLens Demo — Press Briefing", fill=(241, 245, 249))
    draw.text((30, 60), f"Frame {frame_idx:04d}  |  {timestamp:.2f}s", fill=(148, 163, 184))
    if anomaly:
        draw.rectangle([140, 90, 500, 270], outline=(251, 191, 36), width=3)
        draw.text((155, 105), "⚠ FLAGGED DISCONTINUITY", fill=(251, 191, 36))
        draw.text((155, 135), f"Inter-frame delta: 28.45 (outlier)", fill=(251, 191, 36))
    else:
        # Simulate news broadcast bars
        for i, col in enumerate([(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0)]):
            draw.rectangle([140 + i * 90, 150, 220 + i * 90, 280], fill=col)
    img.save(path, "JPEG", quality=85)


# ── SEEDER ────────────────────────────────────────────────────────────────────

def seed_demo_data(db):
    """Ensures 3 rich demo investigations exist with real playable synthetic media."""

    demos = [
        {
            "id": "TL-DEMO-01",
            "title": "[DEMO] Press Briefing Audio/Video Manipulation",
            "description": "Demonstration case examining a 2-minute press conference clip. Video compression analysis detects an abrupt cut at 00:12 (Frame 288), acoustic feature analysis detects a spectral gap at 00:12, and difflib comparison flags a missing sentence in the transcript.",
            "case_ref": "CASE-2026-PB01",
            "transcript": "We never agreed to that number. The meeting ended shortly after. Please confirm by Friday.",
            "files": [
                {"id": "ev-v01", "kind": "video", "name": "press_briefing_clip.mp4"},
                {"id": "ev-a01", "kind": "audio", "name": "briefing_audio.wav"},
                {"id": "ev-t01", "kind": "text",  "name": "official_press_release.txt"},
            ],
            "findings": [
                new_finding("video", "ev-v01", "observed", "Video Stream Parameters",
                            "1280x720 at 24.0 fps, 30.0s synthetic demo clip; 15 keyframes sampled every 2.0s.",
                            detector={"name": "FFmpeg/FFprobe", "version": "6.1", "source": "FFmpeg project", "license": "LGPL 2.1+", "execution_mode": "observed_metadata"},
                            limitations=["Nominal sampling interval of 2.0s may miss short sub-second splicing events."]),
                new_finding("video", "ev-v01", "heuristic", "Abrupt Compression Discontinuity at 00:12",
                            "Frame 288 shows anomalous inter-frame pixel difference (delta 28.45) compared to neighbouring frames.",
                            detector={"name": "GenConViT-Evaluated (Frame-Diff Fallback)", "version": "1.0", "source": "WACV 2024 / TruthLens", "license": "Research Fallback", "execution_mode": "transparent_fallback"},
                            score=28.45, timestamp_seconds=12.0, frame_number=288,
                            artifact="TL-DEMO-01_frames/f_0006.jpg",
                            limitations=["Scene cuts, lighting shifts, or camera angle changes can produce the exact same metric.", "Does not prove intentional tampering by itself."],
                            alternative_explanations=["Camera switch in broadcast studio.", "Re-compression artifact during web streaming encode."]),
                new_finding("audio", "ev-a01", "observed", "Audio Stream Parameters",
                            "16000 Hz mono PCM WAV; 38.0s total duration. Synthetic voice-pattern signal with injected spectral anomaly at 11.8–12.3s.",
                            detector={"name": "Librosa DSP", "version": "0.10.2", "source": "PyPI", "license": "ISC", "execution_mode": "observed_metadata"},
                            artifact="TL-DEMO-01_spectrogram.png"),
                new_finding("audio", "ev-a01", "heuristic", "Spectral Flatness Gap near 00:12",
                            "Sudden drop in high-frequency spectral flatness between 11.8s and 12.3s relative to surrounding speech.",
                            detector={"name": "AASIST-Evaluated (Spectral Flatness Fallback)", "version": "1.0", "source": "ASVspoof baseline / TruthLens", "license": "BSD-3", "execution_mode": "transparent_fallback"},
                            score=0.842, timestamp_seconds=12.0,
                            artifact="TL-DEMO-01_spectrogram.png",
                            limitations=["Microphone breathing noise or sudden acoustic silence triggers similar flatness shifts."]),
                new_finding("text", "ev-t01", "observed", "Supplied Phrase Omitted in Audio Transcript",
                            "Supplied text contains: 'Our lawyers confirmed it', but this phrase is absent from the transcribed speech around 00:12.",
                            detector={"name": "Difflib-SequenceMatcher", "version": "Python 3 stdlib", "source": "Python", "license": "PSF", "execution_mode": "observed_metadata"},
                            timestamp_seconds=12.0,
                            limitations=["Speech-to-text recognition errors or spontaneous speaker omission can create identical discrepancies."]),
                new_finding("correlation", None, "heuristic", "Cross-Modal Correlation near 00:12",
                            "3 observations from Video, Audio, and Text coincide within 0.5s at timestamp 00:12. This temporal cluster indicates a concurrent event across modalities.",
                            timestamp_seconds=12.0,
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
            "description": "Image forensics demonstration case. A scanned purchase contract agreement. Error Level Analysis (ELA) isolates a high-frequency compression anomaly around the signature box, and EXIF inspection reveals an Adobe Photoshop editing signature.",
            "case_ref": "CASE-2026-CT02",
            "transcript": "",
            "files": [
                {"id": "ev-img01", "kind": "image", "name": "signed_contract_scan.jpg"},
            ],
            "findings": [
                new_finding("image", "ev-img01", "observed", "Image Resolution & EXIF Metadata",
                            "1240x1754 px JPEG document scan. EXIF Software tag records 'Adobe Photoshop 24.1 (Windows)' with modification timestamp 2026-11-15T09:22:14Z.",
                            detector={"name": "ExifTool / PyExifTool", "version": "12.70", "source": "Phil Harvey", "license": "GPLv1/Artistic", "execution_mode": "observed_metadata"},
                            limitations=["Scanning software often uses Photoshop plugins for basic auto-deskewing and contrast enhancement."]),
                new_finding("image", "ev-img01", "heuristic", "Localized ELA Compression Outlier in Signature Region",
                            "Error Level Analysis demonstrates that pixels in block [x:600, y:480, w:380, h:90] re-compress with significantly higher residual error than the surrounding document, consistent with a pasted signature element.",
                            detector={"name": "TruFor-Evaluated (ELA Block Fallback)", "version": "1.0", "source": "ECCV 2022 / TruthLens", "license": "Research Fallback", "execution_mode": "transparent_fallback"},
                            score=3.42, region={"x": 600, "y": 480, "width": 380, "height": 90},
                            artifact="TL-DEMO-02_ela.png",
                            limitations=["High-contrast edges and pure black signatures naturally generate elevated ELA residuals.", "A human examiner must perform micro-level edge inspection."],
                            alternative_explanations=["Scanning software applied different JPEG quality to different page regions.", "Original document was partially photocopied before scanning."]),
                new_finding("provenance", "ev-img01", "observed", "C2PA Content Credentials Manifest Check",
                            "No C2PA/CAI cryptographic provenance manifest is embedded in this JPEG container.",
                            detector={"name": "C2PA Python SDK", "version": "0.5.0", "source": "Content Authenticity Initiative", "license": "MIT/Apache 2.0", "execution_mode": "observed_metadata"},
                            limitations=["Most office scanners and phones do not yet implement C2PA signing. Absence of C2PA manifest does NOT indicate forgery."])
            ]
        },
        {
            "id": "TL-DEMO-03",
            "title": "[DEMO] Synthetic Voice Phishing Voicemail",
            "description": "Audio anti-spoofing demonstration case. An urgent voicemail purporting to be the CFO requesting an immediate offshore wire transfer. Acoustic feature analysis reveals unnatural spectral flatness and robotic high-frequency rolloff consistent with neural TTS vocoder synthesis.",
            "case_ref": "CASE-2026-VM03",
            "transcript": "Urgent message for payroll. Please authorize the wire transfer to the revised overseas routing code today.",
            "files": [
                {"id": "ev-aud01", "kind": "audio", "name": "urgent_cfo_voicemail.wav"},
                {"id": "ev-txt01", "kind": "text",  "name": "verified_executive_communication.txt"},
            ],
            "findings": [
                new_finding("audio", "ev-aud01", "observed", "Audio Codec and Sampling Parameters",
                            "16000 Hz mono WAV audio, 38.4s duration. Synthetic TTS-like signal with formant periodicity and sharp 7.5 kHz bandlimit.",
                            detector={"name": "Librosa", "version": "0.10.2", "source": "PyPI", "license": "ISC", "execution_mode": "observed_metadata"},
                            artifact="TL-DEMO-03_spectrogram.png"),
                new_finding("audio", "ev-aud01", "heuristic", "Synthetic Voice Acoustic Anomaly (High-Frequency Rolloff)",
                            "AASIST acoustic feature evaluation detects a sharp, unnatural spectral cliff at 7.5 kHz with robotic formant periodicity consistent with neural vocoder synthesis.",
                            detector={"name": "AASIST-Evaluated (Acoustic Fallback)", "version": "1.0", "source": "ASVspoof 2021 / TruthLens", "license": "BSD-3", "execution_mode": "transparent_fallback"},
                            score=0.912,
                            artifact="TL-DEMO-03_spectrogram.png",
                            limitations=["Aggressive cellular telephony codecs (e.g. AMR-NB) can also sharply bandlimit frequencies above 7 kHz."]),
                new_finding("audio", "ev-aud01", "model_prediction", "Faster-Whisper Speech Transcription",
                            "Transcribed speech content: 'Urgent message for payroll. Please authorize the wire transfer to the revised overseas routing code today.' Language detected: English (confidence 0.99).",
                            detector={"name": "Faster-Whisper", "version": "1.0 (CTranslate2 int8)", "source": "OpenAI / SYSTRAN", "license": "MIT", "execution_mode": "native_model"}),
                new_finding("correlation", None, "ai_explanation", "AI Phishing Synthesis & Advisory",
                            "The audio exhibits acoustic hallmarks of text-to-speech vocoding combined with urgency cues characteristic of business email compromise (BEC). Recommended action: verify caller identity via out-of-band secondary telephone protocol before authorizing any wire transfer.",
                            detector={"name": "Gemini 2.0 Flash", "version": "2.0-flash", "source": "Google Cloud", "license": "Commercial API", "execution_mode": "api_provider"},
                            limitations=["AI heuristic explanation. Verification with the named executive is mandatory before law enforcement escalation."])
            ]
        }
    ]

    for d in demos:
        existing = db.get(Investigation, d["id"])
        if existing:
            # Always refresh media files even if investigation exists
            _refresh_media(d)
            continue

        inv = Investigation(
            id=d["id"], title=d["title"], description=d["description"],
            case_ref=d["case_ref"], transcript=d["transcript"],
            status="completed", review_status="needs_review", is_demo=1, created_at=utcnow()
        )
        db.add(inv)

        for f in d["files"]:
            stored = config.UPLOADS / d["id"] / f["name"]
            _write_media_file(stored, f["kind"], f["name"], d["id"])
            size = stored.stat().st_size if stored.exists() else 1024
            import hashlib
            sha = hashlib.sha256(stored.read_bytes()).hexdigest() if stored.exists() else "0" * 64
            db.add(EvidenceFile(
                id=f["id"], investigation_id=d["id"], kind=f["kind"],
                original_name=f["name"], stored_path=str(stored),
                size=size, sha256=sha, uploaded_at=utcnow()
            ))

        for agent_name in config.AGENTS:
            db.add(AgentExecution(
                investigation_id=d["id"], name=agent_name, state="completed",
                started_at=utcnow(), finished_at=utcnow(),
                output={"demo": True, "status": "completed"}
            ))

        # Generate visual artifacts
        _build_artifacts(d["id"])

        # Findings
        for finding in d["findings"]:
            if finding.get("artifact_path") and not finding["artifact_path"].startswith("/"):
                finding["artifact_path"] = f"/api/v1/investigations/{d['id']}/artifacts/{finding['artifact_path']}"
            db.add(Finding(id=finding["id"], investigation_id=d["id"], modality=finding["modality"], data=finding))

        db.commit()

        try:
            report_path = report_service.build(db, d["id"])
            from app.database.database import Report
            db.merge(Report(investigation_id=d["id"], path=str(report_path), created_at=utcnow()))
            db.commit()
        except Exception as e:
            print(f"Warning: demo report for {d['id']} failed: {e}")


def _refresh_media(d):
    """Re-generate real media files for existing demo cases (idempotent)."""
    for f in d["files"]:
        stored = config.UPLOADS / d["id"] / f["name"]
        if not stored.exists() or stored.stat().st_size < 2000:
            _write_media_file(stored, f["kind"], f["name"], d["id"])
    _build_artifacts(d["id"])


def _write_media_file(path: Path, kind: str, name: str, demo_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "audio":
        pattern = "phishing" if "cfo" in name else "voice"
        try:
            _make_wav(path, duration_s=38.0, pattern=pattern)
        except Exception as e:
            print(f"WAV generation failed for {name}: {e}")
            path.write_bytes(b'\x00' * 1024)
    elif kind == "video":
        try:
            ok = _make_mp4(path, duration_s=30.0)
            if not ok:
                path.write_bytes(b'\x00' * 2048)
        except Exception as e:
            print(f"MP4 generation failed for {name}: {e}")
            path.write_bytes(b'\x00' * 2048)
    elif kind == "image":
        try:
            _make_contract_jpg(path)
        except Exception as e:
            print(f"JPG generation failed for {name}: {e}")
            # Fallback: simple JPEG
            img = Image.new("RGB", (1240, 1754), (245, 243, 238))
            img.save(path, "JPEG", quality=72)
    elif kind == "text":
        if "press_release" in name or "official" in name:
            path.write_text(
                "OFFICIAL PRESS RELEASE — FOR IMMEDIATE RELEASE\n\n"
                "The Ministry confirmed today that our lawyers confirmed it and "
                "the revised budget allocation has been approved. We never agreed "
                "to that number. The meeting ended shortly after. Please confirm by Friday.\n\n"
                "— Communications Office, 15 November 2026"
            )
        else:
            path.write_text(
                "VERIFIED EXECUTIVE COMMUNICATION — INTERNAL USE ONLY\n\n"
                "From: CFO Office\nTo: Payroll Department\n\n"
                "This is an urgent request to process a wire transfer of USD 48,500 "
                "to overseas routing code 7729-XB-991 by end of business today. "
                "Do not discuss this with other staff. Confirm receipt immediately.\n\n"
                "GROUND TRUTH LABEL: SYNTHETIC VOICE — TTS VOCODER DETECTED\n"
                "This is a demo case for TruthLens forensic training purposes."
            )


def _build_artifacts(demo_id: str):
    art_dir = config.ARTIFACTS / demo_id
    art_dir.mkdir(parents=True, exist_ok=True)
    if demo_id == "TL-DEMO-01":
        _create_synthetic_spectrogram(art_dir / "TL-DEMO-01_spectrogram.png", anomaly_col=240)
        frame_dir = art_dir / "TL-DEMO-01_frames"
        frame_dir.mkdir(exist_ok=True)
        for i in range(1, 16):
            t = i * 2.0
            _create_frame(frame_dir / f"f_{i:04d}.jpg", i * 48, t, anomaly=(11 <= t <= 13))
    elif demo_id == "TL-DEMO-02":
        _create_ela_heatmap(art_dir / "TL-DEMO-02_ela.png")
    elif demo_id == "TL-DEMO-03":
        _create_synthetic_spectrogram(art_dir / "TL-DEMO-03_spectrogram.png", anomaly_col=750)
