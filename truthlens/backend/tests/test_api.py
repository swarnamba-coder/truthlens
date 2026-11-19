import io, numpy as np
from PIL import Image

def _jpg():
    b = io.BytesIO()
    Image.fromarray(np.random.default_rng(2).integers(0, 255, (200, 200, 3), dtype=np.uint8)).save(b, "JPEG")
    return b.getvalue()

def test_health(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"

def test_rejects_unsupported_and_empty(client):
    r = client.post("/api/v1/investigations", data={"title": "t"}, files={"image": ("x.exe", b"MZ")})
    assert r.status_code == 422 and "error" in r.json()
    assert client.post("/api/v1/investigations", data={"title": "t"}).status_code == 422

def test_full_flow_with_corrupt_file_reported_honestly(client):
    r = client.post("/api/v1/investigations", data={"title": "Case", "transcript": "hello there friend"},
                    files=[("image", ("a.jpg", _jpg())), ("image", ("bad.jpg", b"junk"))])
    assert r.status_code == 201
    inv = r.json(); assert all(len(e["sha256"]) == 64 for e in inv["evidence"])
    assert client.post(f"/api/v1/investigations/{inv['id']}/analyze").status_code == 202
    st = client.get(f"/api/v1/investigations/{inv['id']}/status").json()
    states = {a["name"]: a["state"] for a in st["agents"]}
    assert states["image"] == "completed" and states["video"] == "unavailable" and states["report"] == "completed"
    res = client.get(f"/api/v1/investigations/{inv['id']}/results").json()["results"]
    assert res["image"]["findings"] and res["text"]["status"] == "unavailable"  # no audio transcript, nothing invented
    pdf = client.get(f"/api/v1/investigations/{inv['id']}/report")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
