from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_check():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json().get("status") == "ok"

def test_verify_text_endpoint_supported():
    payload = {"claim": "This is definitely true"}
    res = client.post("/api/verify/text", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "verdict" in data
    assert "score" in data
