# tests/test_health.py
from fastapi.testclient import TestClient
from drone_sys.app.main import app

client = TestClient(app)


def test_ping():
    resp = client.get("/health/ping")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"