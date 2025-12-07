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

def test_verify_text_non_factual():
    res = client.post("/api/verify/text", json={"claim": "I want to go to India"})
    assert res.status_code == 200
    data = res.json()
    # Non-factual claims should not trigger Wikipedia evidence
    assert data["evidence"] == []


def test_verify_text_simple_factual():
    res = client.post("/api/verify/text", json={"claim": "India is in South Asia"})
    assert res.status_code == 200
    data = res.json()
    # We at least expect some evidence array,
    # even if verdict is NOT_ENOUGH_INFO or SUPPORTED depending on similarity.
    assert "evidence" in data
    assert isinstance(data["evidence"], list)

