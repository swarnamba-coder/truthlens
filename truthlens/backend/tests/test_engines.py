import numpy as np
from PIL import Image
from app.engines import image_engine, text_engine
from app.correlation.correlation_engine import correlate

def test_image_valid_and_corrupt(tmp_path):
    p = tmp_path / "a.jpg"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(p)
    assert image_engine.analyze(p, "i1", tmp_path)["status"] == "completed"
    bad = tmp_path / "b.jpg"; bad.write_bytes(b"nope")
    assert image_engine.analyze(bad, "i2", tmp_path)["status"] == "failed"

def test_text_matching_and_missing_phrase():
    segs = [{"start": 0.0, "text": "we never agreed to that number"}]
    same = text_engine.compare("We never agreed to that number.", segs)
    assert same["status"] == "completed" and same["findings"] == []
    diff = text_engine.compare("we never agreed to that number and our lawyers confirmed it", segs)
    assert any("not found in audio" in f["title"] for f in diff["findings"])

def test_text_unavailable_without_audio():
    assert text_engine.compare("hello there", [])["status"] == "unavailable"

def test_correlation_links_only_across_modalities():
    fs = [{"id": "1", "modality": "video", "timestamp_seconds": 12.0}, {"id": "2", "modality": "audio", "timestamp_seconds": 12.5},
          {"id": "3", "modality": "audio", "timestamp_seconds": 60.0}, {"id": "4", "modality": "image", "timestamp_seconds": None}]
    out = correlate(fs, 2.0)
    assert len(out["findings"]) == 1 and set(out["findings"][0]["related_finding_ids"]) == {"1", "2"}
    assert set(out["metadata"]["independent_finding_ids"]) == {"3", "4"}
