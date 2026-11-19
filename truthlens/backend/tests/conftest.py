import os, tempfile
_d = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _d
os.environ["DATABASE_URL"] = f"sqlite:///{_d}/test.db"
os.environ["GEMINI_ENABLED"] = "false"
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
