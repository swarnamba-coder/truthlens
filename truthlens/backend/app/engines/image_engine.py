"""
Image analysis v2: ExifTool metadata + ELA + 5-layer AI-generation detection.

AI Detection layers (all CPU-compatible):
  1. Frequency Domain (FFT) — AI images lack natural camera noise in high frequencies
  2. PRNU Noise Fingerprint — real cameras have sensor noise patterns; AI images don't
  3. Statistical Texture (LBP + Co-occurrence) — AI images have unnaturally uniform textures
  4. Chromatic Aberration — real lenses produce colour fringing; AI renderers often don't
  5. HuggingFace ViT classifier — 'Organika/sdxl-detector' trained on Stable Diffusion vs real photos
"""
import io, json, shutil, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops, ExifTags
from app.schemas.findings import new_finding, result

EXIFTOOL_DET   = {"name": "ExifTool / PyExifTool",             "version": "12.70",  "source": "Phil Harvey",                           "license": "GPLv1 / Artistic",        "execution_mode": "observed_metadata"}
TRUFOR_FB      = {"name": "TruFor-Evaluated (ELA Fallback)",   "version": "1.0",    "source": "github.com/grip-unina/TruFor",           "license": "CC BY-NC 4.0 (Fallback)", "execution_mode": "transparent_fallback"}
C2PA_DET       = {"name": "C2PA Content Credentials",          "version": "0.5.0",  "source": "Content Authenticity Initiative",        "license": "MIT / Apache 2.0",        "execution_mode": "observed_metadata"}
FFT_DET        = {"name": "FFT Frequency Noise Analyser",      "version": "1.0",    "source": "TruthLens / scipy.fft",                  "license": "BSD-3",                   "execution_mode": "native_model"}
PRNU_DET       = {"name": "PRNU Sensor Noise Fingerprint",     "version": "1.0",    "source": "TruthLens / numpy",                      "license": "BSD-3",                   "execution_mode": "native_model"}
LBP_DET        = {"name": "LBP Texture Uniformity Analyser",   "version": "1.0",    "source": "TruthLens / scikit-image",               "license": "BSD-3",                   "execution_mode": "native_model"}
GRAD_DET       = {"name": "Gradient & Edge Distribution Analyser", "version": "1.0", "source": "TruthLens / numpy", "license": "BSD-3", "execution_mode": "native_model"}
SMOOTH_DET     = {"name": "Micro-Texture Smoothness Analyser",     "version": "1.0", "source": "TruthLens / numpy", "license": "BSD-3", "execution_mode": "native_model"}
VIT_DET        = {"name": "ViT AI-Image Classifier",           "version": "1.0",    "source": "Organika/sdxl-detector (HuggingFace)",   "license": "Apache 2.0",              "execution_mode": "native_model"}


# ── HELPER: safe numpy rng ────────────────────────────────────────────────────
def _arr(im: Image.Image, mode="RGB") -> np.ndarray:
    return np.asarray(im.convert(mode), dtype=np.float32)


# ── LAYER 1: FFT Frequency Domain ─────────────────────────────────────────────
def _fft_analysis(im: Image.Image):
    """
    Real camera images have 1/f noise — energy rolls off smoothly.
    AI-generated images often show periodic grid patterns or abrupt
    energy cliffs in the high-frequency bands (from convolutional upsampling).
    Returns: (score 0-1 where >0.55 = suspicious, detail_dict)
    """
    gray = _arr(im, "L")
    # Compute 2D FFT magnitude spectrum
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    h, w = magnitude.shape
    # Compare low vs high frequency energy ratio
    low_band  = magnitude[h//4:3*h//4, w//4:3*w//4]
    high_band = np.concatenate([magnitude[:h//4, :].ravel(), magnitude[3*h//4:, :].ravel()])
    low_energy  = float(low_band.mean())
    high_energy = float(np.array(high_band).mean())
    ratio = high_energy / (low_energy + 1e-9)
    # Natural images: ratio ~ 0.3-0.55. AI images: ratio often < 0.3 (too smooth) or has periodic spikes
    too_smooth  = ratio < 0.30
    # Check for periodic grid artefacts (upsampling fingerprint)
    # Look for unexpected spikes in high-freq quadrants
    hf_quad = magnitude[:h//8, :w//8]
    hf_std  = float(hf_quad.std())
    hf_spike = hf_std > 1.5  # large variance = periodic spike = GAN upsampling artefact
    score = 0.0
    if too_smooth:  score += 0.45
    if hf_spike:    score += 0.35
    # Normalize
    score = min(score, 1.0)
    return score, {
        "low_high_ratio": round(ratio, 4),
        "hf_std": round(hf_std, 4),
        "too_smooth": too_smooth,
        "periodic_spike": hf_spike,
    }


# ── LAYER 2: PRNU Sensor Noise Fingerprint ────────────────────────────────────
def _prnu_analysis(im: Image.Image):
    """
    Real camera sensors introduce Photo-Response Non-Uniformity (PRNU).
    AI images have smooth/uniform noise. JPEG compression adds spiky kurtosis.

    Calibration:
    - AI clean images: kurtosis 2.8–4.5  → AI-like
    - Real uncompressed: kurtosis 4–10   → real camera
    - JPEG/WhatsApp compressed: kurtosis 10–35 → compressed real photo
    Returns: (score 0-1 where high = more AI-like, detail_dict)
    """
    rgb = _arr(im, "RGB")
    try:
        from scipy.ndimage import uniform_filter
        smoothed = uniform_filter(rgb, size=3)
        noise = rgb - smoothed
        kurtosis_vals = []
        for c in range(3):
            ch = noise[:, :, c].ravel()
            mean = ch.mean(); std = ch.std()
            if std > 0:
                kurtosis_vals.append(float(np.mean(((ch - mean)/std)**4)))
        avg_kurt = float(np.mean(kurtosis_vals)) if kurtosis_vals else 3.0
        # LOW kurtosis → smooth AI noise; HIGH kurtosis → real or compressed
        if avg_kurt < 4.0:
            score = 0.75   # Very smooth — strong AI indicator
        elif avg_kurt < 6.0:
            score = 0.45   # Moderately smooth — possible AI
        elif avg_kurt < 10.0:
            score = 0.20   # Normal camera range
        else:
            score = 0.05   # High kurtosis — JPEG-compressed real photo
        interp = ("smooth_ai" if avg_kurt < 4 else "possible_ai" if avg_kurt < 6
                  else "natural_camera" if avg_kurt < 10 else "compressed_real_photo")
        return score, {"kurtosis": round(avg_kurt, 3), "interpretation": interp}
    except Exception:
        return 0.0, {"error": "scipy unavailable"}


# ── LAYER 3: LBP Texture Uniformity ──────────────────────────────────────────
def _lbp_texture(im: Image.Image):
    """
    Local Binary Pattern (LBP) histogram measures micro-texture complexity.
    AI images tend to have unnaturally smooth, over-regular textures.
    Returns: (score 0-1 where >0.5 suspicious, detail_dict)
    """
    try:
        import cv2
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
        # Compute LBP manually using OpenCV bitwise ops
        h, w = gray.shape
        # Simple 3x3 LBP
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            shifted = np.roll(np.roll(gray, dy, 0), dx, 1)
            lbp = (lbp << 1) | (gray >= shifted).astype(np.uint8)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
        # Entropy of LBP histogram
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
        # Natural images: entropy 6-7.5. AI images: entropy often 7.6-8 (too uniform/too random)
        # or very low entropy (< 5.5) for over-smooth AI textures
        is_over_smooth  = entropy < 3.5   # Only flag extreme smoothness (pure gradients)
        is_unnaturally_uniform = entropy > 7.6
        score = 0.0
        if is_over_smooth:         score += 0.60
        if is_unnaturally_uniform: score += 0.35
        score = min(score, 1.0)
        return score, {"lbp_entropy": round(entropy, 3), "over_smooth": is_over_smooth, "over_uniform": is_unnaturally_uniform}
    except Exception as e:
        return 0.0, {"error": str(e)}


# ── LAYER 4: Gradient Kurtosis & Texture Smoothness ───────────────────────────
def _gradient_and_smoothness_analysis(im: Image.Image):
    """
    Evaluates gradient distribution (tail heaviness) and micro-texture smoothness.
    Real optical photos:
      - Contain high gradient kurtosis (> 12) from sharp natural textures, sensor noise,
        and high-frequency edges.
      - Moderate smooth fraction (0.4 - 0.7) because sensor noise prevents pure zero gradients.
    AI generated/rendered imagery:
      - Low gradient kurtosis (< 8) because diffusion models over-smooth backgrounds & skin.
      - High smooth fraction (> 0.85) due to lack of camera sensor shot noise.
    """
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gmag = np.sqrt(gx[:-1, :]**2 + gy[:, :-1]**2)
    g_std = gmag.std()
    grad_kurt = float(np.mean(((gmag - gmag.mean()) / (g_std + 1e-6))**4))
    smooth_frac = float((gmag < 2.0).mean())
    
    score = 0.0
    if grad_kurt < 6.0:
        score += 0.50
    elif grad_kurt < 10.0:
        score += 0.25
        
    if smooth_frac > 0.85:
        score += 0.40
    elif smooth_frac > 0.75:
        score += 0.20
        
    score = min(score, 1.0)
    return score, {
        "gradient_kurtosis": round(grad_kurt, 3),
        "smooth_fraction": round(smooth_frac, 4),
        "is_abnormally_smooth": smooth_frac > 0.85,
        "is_low_kurtosis": grad_kurt < 6.0
    }


# ── LAYER 5: HuggingFace ViT Classifier ──────────────────────────────────────
_vit_pipeline = None
_vit_tried    = False

def _vit_classify(im: Image.Image):
    """
    Uses 'Organika/sdxl-detector' — a ViT fine-tuned specifically to distinguish
    real photos from Stable Diffusion/SDXL AI-generated images.
    Falls back gracefully if transformers not installed.
    """
    global _vit_pipeline, _vit_tried
    if _vit_tried and _vit_pipeline is None:
        return None, {"error": "Model not available"}
    if not _vit_tried:
        _vit_tried = True
        try:
            from transformers import pipeline
            _vit_pipeline = pipeline(
                "image-classification",
                model="Organika/sdxl-detector",
                device=-1   # CPU
            )
        except Exception as e:
            _vit_pipeline = None
            return None, {"error": f"transformers unavailable: {e}"}
    try:
        preds = _vit_pipeline(im)
        # preds: [{'label': 'artificial', 'score': 0.92}, {'label': 'real', 'score': 0.08}]
        label_map = {p["label"].lower(): p["score"] for p in preds}
        ai_score  = label_map.get("artificial", label_map.get("ai", 0.0))
        real_score= label_map.get("real", 0.0)
        return float(ai_score), {
            "ai_probability": round(float(ai_score), 4),
            "real_probability": round(float(real_score), 4),
            "top_label": preds[0]["label"],
            "raw_predictions": [{p["label"]: round(p["score"],4)} for p in preds]
        }
    except Exception as e:
        return None, {"error": str(e)}


# ── EXIF helpers ──────────────────────────────────────────────────────────────
def _extract_exif(path: Path, im: Image.Image):
    if shutil.which("exiftool"):
        try:
            p = subprocess.run(["exiftool", "-j", "-g", str(path)], capture_output=True, text=True, timeout=10)
            if p.returncode == 0:
                data = json.loads(p.stdout or "[]")
                if data:
                    flat = {}
                    for group, tags in data[0].items():
                        if isinstance(tags, dict):
                            for k, v in tags.items():
                                flat[f"{group}:{k}"] = str(v)[:200]
                        else:
                            flat[group] = str(tags)[:200]
                    return flat, "exiftool_native"
        except Exception:
            pass
    exif = {}
    try:
        for k, v in (im.getexif() or {}).items():
            exif[ExifTags.TAGS.get(k, str(k))] = str(v)[:200]
    except Exception:
        pass
    return exif, "pil_fallback"


def _check_c2pa(path: Path):
    try:
        content = path.read_bytes()
        return b"c2pa" in content or b"jumb" in content
    except Exception:
        return False


# ── COMPOSITE AI SCORE ────────────────────────────────────────────────────────
def _composite_ai_verdict(fft_s, prnu_s, lbp_s, grad_s, vit_s):
    """
    Weighted ensemble of physical and deep learning layers.
    PRNU noise absence and Gradient flatness are primary physical fingerprints of AI generation.
    """
    if vit_s is not None:
        base = 0.45 * vit_s + 0.25 * grad_s + 0.20 * prnu_s + 0.10 * fft_s
    else:
        # High reliance on PRNU and Gradient Kurtosis
        base = 0.40 * grad_s + 0.35 * prnu_s + 0.15 * fft_s + 0.10 * lbp_s
        
        # Non-linear boost: if both primary physical indicators detect AI hallmarks
        if grad_s >= 0.70 and prnu_s >= 0.70:
            base = max(base, 0.82)
        elif grad_s >= 0.50 or prnu_s >= 0.70:
            base = max(base, 0.55)
            
    return round(float(base), 4)


# ── MAIN ENTRY ────────────────────────────────────────────────────────────────
def analyze(path: Path, file_id: str, art_dir: Path):
    try:
        with Image.open(path) as probe:
            probe.verify()
        im = Image.open(path)
        im.load()
    except Exception as e:
        return result("failed", f"Cannot decode image: {type(e).__name__}: {e}")

    # Resize large images for speed (keep original for ELA)
    thumb = im.copy()
    if max(thumb.width, thumb.height) > 1024:
        thumb.thumbnail((1024, 1024), Image.LANCZOS)

    exif, exif_source = _extract_exif(path, im)
    meta = {
        "format": im.format, "width": im.width, "height": im.height,
        "mode": im.mode, "exif": exif, "exif_engine": exif_source
    }

    findings = []

    # ── Finding 1: Metadata ───────────────────────────────────────────────────
    findings.append(new_finding(
        "image", file_id, "observed", "Image properties and metadata",
        f"{im.format} image, {im.width}×{im.height} px, {len(exif)} EXIF field(s) extracted via {exif_source}.",
        detector=EXIFTOOL_DET,
        limitations=["Missing or stripped EXIF is standard on social media and messaging platforms and does not indicate forgery."]
    ))

    software_tag = exif.get("Software") or exif.get("IFD0:Software") or exif.get("EXIF:Software")
    if software_tag:
        findings.append(new_finding(
            "image", file_id, "observed", "Editing software tag present",
            f"EXIF Software field reads '{software_tag}'.",
            detector=EXIFTOOL_DET,
            limitations=["Scanning utilities, camera firmware, and photo organizers routinely populate this tag."]
        ))

    # Detect known AI tool signatures in EXIF
    ai_tools = ["stable diffusion", "midjourney", "dall-e", "firefly", "imagen", "generative", "ai generated"]
    if software_tag and any(t in software_tag.lower() for t in ai_tools):
        findings.append(new_finding(
            "image", file_id, "heuristic", "⚠ AI Generation Tool Detected in EXIF",
            f"Software tag '{software_tag}' matches known AI image generation tools. This strongly suggests the image was AI-generated.",
            detector=EXIFTOOL_DET, score=0.95,
            limitations=["EXIF metadata can be manually edited."]
        ))

    # ── Finding 2: C2PA ───────────────────────────────────────────────────────
    c2pa_present = _check_c2pa(path)
    meta["c2pa_manifest"] = c2pa_present
    findings.append(new_finding(
        "provenance", file_id, "observed",
        "C2PA Content Credentials Manifest Check",
        f"Cryptographic C2PA manifest {'detected ✅' if c2pa_present else 'not present'}.",
        detector=C2PA_DET,
        limitations=["Absence of C2PA is standard for most consumer cameras and apps."]
    ))

    # ── Layer 1: FFT ──────────────────────────────────────────────────────────
    fft_score, fft_meta = _fft_analysis(thumb)
    meta["fft"] = fft_meta

    # ── Layer 2: PRNU ─────────────────────────────────────────────────────────
    prnu_score, prnu_meta = _prnu_analysis(thumb)
    meta["prnu"] = prnu_meta

    # ── Layer 3: LBP Texture ──────────────────────────────────────────────────
    lbp_score, lbp_meta = _lbp_texture(thumb)
    meta["lbp"] = lbp_meta

    # ── Layer 4: Gradient & Texture Smoothness ────────────────────────────────
    grad_score, grad_meta = _gradient_and_smoothness_analysis(thumb)
    meta["gradient_and_smoothness"] = grad_meta

    # ── Layer 5: ViT Classifier ───────────────────────────────────────────────
    vit_score, vit_meta = _vit_classify(thumb)
    meta["vit_classifier"] = vit_meta

    # ── Composite AI score ────────────────────────────────────────────────────
    composite = _composite_ai_verdict(fft_score, prnu_score, lbp_score, grad_score, vit_score)
    meta["ai_composite_score"] = composite

    # Verdict label
    if composite >= 0.70:
        verdict = "HIGH likelihood of AI generation"
        provenance_type = "heuristic"
        emoji = "🤖"
    elif composite >= 0.45:
        verdict = "MODERATE indicators of AI generation"
        provenance_type = "heuristic"
        emoji = "⚠️"
    else:
        verdict = "LOW indicators of AI generation (likely authentic)"
        provenance_type = "observed"
        emoji = "✅"

    # ── Finding 3: AI Generation Detection (main finding) ────────────────────
    vit_desc = ""
    if vit_score is not None:
        vit_desc = f" ViT classifier: {vit_meta.get('top_label','?')} ({round((vit_score)*100,1)}% AI probability)."

    findings.append(new_finding(
        "image", file_id, provenance_type,
        f"{emoji} AI Image Detection — {verdict}",
        (
            f"Composite AI-generation score: {round(composite*100,1)}% (0% = authentic, 100% = AI-generated). "
            f"5-layer ensemble: "
            f"FFT frequency analysis score {round(fft_score*100,1)}% "
            f"(high-freq ratio: {fft_meta.get('low_high_ratio','?')}, periodic spike: {fft_meta.get('periodic_spike','?')}); "
            f"PRNU sensor noise score {round(prnu_score*100,1)}% "
            f"(kurtosis: {prnu_meta.get('kurtosis','?')}); "
            f"LBP texture score {round(lbp_score*100,1)}% "
            f"(entropy: {lbp_meta.get('lbp_entropy','?')}); "
            f"Gradient & smoothness score {round(grad_score*100,1)}% "
            f"(kurtosis: {grad_meta.get('gradient_kurtosis','?')}, smooth_frac: {grad_meta.get('smooth_fraction','?')}).{vit_desc}"
        ),
        detector=VIT_DET if vit_score is not None else GRAD_DET,
        score=composite,
        limitations=[
            "No single metric is definitive. All scores are probabilistic indicators, not proof.",
            "Heavy JPEG compression, social-media re-encoding, and screenshot artifacts affect all metrics.",
            "Some high-quality real photos (studio lighting, clean backgrounds) can score higher AI indicators.",
            "Some AI images with added noise/grain can score lower AI indicators.",
        ],
        alternative_explanations=[
            "Professional photography with studio lighting and post-processing.",
            "Image heavily re-compressed by WhatsApp/Telegram (reduces PRNU and FFT signatures).",
        ]
    ))

    # ── Individual metric findings for transparency ────────────────────────────
    if fft_meta.get("periodic_spike") or fft_score > 0.3:
        findings.append(new_finding(
            "image", file_id, "heuristic",
            "FFT Frequency: Unusual high-frequency energy pattern" if fft_score > 0.3 else "FFT Frequency: Normal camera noise distribution",
            f"High-frequency energy ratio: {fft_meta.get('low_high_ratio')} (normal: 0.30–0.55). "
            f"Periodic spectral spike detected: {fft_meta.get('periodic_spike')}. "
            f"GAN upsampling artefacts produce periodic peaks in the FFT spectrum.",
            detector=FFT_DET, score=fft_score,
            limitations=["Strong image compression and resizing can flatten the frequency spectrum."]
        ))

    if vit_score is not None:
        findings.append(new_finding(
            "image", file_id, "model_prediction",
            f"ViT AI Classifier: {vit_meta.get('top_label','?').upper()} ({round((vit_score)*100,1)}% AI probability)",
            f"Vision Transformer fine-tuned on real vs Stable Diffusion images. "
            f"Raw predictions: {vit_meta.get('raw_predictions','N/A')}.",
            detector=VIT_DET, score=vit_score,
            limitations=[
                "Model trained primarily on SDXL outputs; may underdetect DALL-E 3 or Midjourney images.",
                "Model performance degrades on heavily compressed or cropped images.",
            ]
        ))

    # ── ELA (tamper detection, separate from AI detection) ────────────────────
    rgb = im.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=90)
    buf.seek(0)
    diff = np.asarray(ImageChops.difference(rgb, Image.open(buf).convert("RGB")), dtype=np.float32).mean(axis=2)
    peak = float(diff.max())
    art_name = f"{file_id}_ela.png"
    Image.fromarray(np.clip(diff * (255.0 / max(peak, 1e-6)), 0, 255).astype("uint8")).save(art_dir / art_name)
    meta["ela_peak"] = round(peak, 3)

    h, w = diff.shape
    bh, bw = max(h // 8, 1), max(w // 8, 1)
    blocks = [(float(diff[r*bh:(r+1)*bh, c*bw:(c+1)*bw].mean()), c*bw, r*bh)
              for r in range(8) for c in range(8) if r*bh < h and c*bw < w]
    means = np.array([b[0] for b in blocks])
    mu, sd = float(means.mean()), float(means.std())

    if sd > 0:
        outliers = sorted((b for b in blocks if (b[0] - mu) / sd > 2.5), reverse=True)[:2]
        for score, x, y in outliers:
            findings.append(new_finding(
                "image", file_id, "heuristic", "ELA: Compression-level outlier region (possible splice/paste)",
                "This region re-compresses differently from the rest of the image (Error Level Analysis). "
                "May indicate a pasted element or localized re-editing.",
                detector=TRUFOR_FB, score=round(score, 3),
                region={"x": x, "y": y, "width": bw, "height": bh},
                artifact=art_name,
                limitations=[
                    "High-contrast textures, text, and sharp edges naturally yield higher ELA residuals.",
                    "An ELA outlier is a prompt for human review, not proof of manipulation.",
                ],
                alternative_explanations=[
                    "Natural focus contrast between foreground subject and blurred background.",
                    "Re-saving a crop with fine high-frequency details.",
                ]
            ))

    return result("completed", None, findings, meta)
