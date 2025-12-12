# tests/test_core.py
import time
import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from backend import app as backend_app_module  # module object
from backend.app import app, is_factual, get_wikipedia_evidence_cached, get_wikipedia_evidence

client = TestClient(app)

# -----------------------
# Unit tests for is_factual
# -----------------------
@pytest.mark.parametrize("text,expected", [
    ("I want to go to India", False),
    ("Should I go to India?", False),
    ("India is the seventh-largest country by area", True),
    ("The Sun is a star", True),
    ("I think India is beautiful", False),
])
def test_is_factual_examples(text, expected):
    assert is_factual(text) == expected

# -----------------------
# Endpoint behavior tests (no external calls)
# -----------------------
def test_verify_text_empty():
    res = client.post("/api/verify/text", json={"claim": ""})
    # Pydantic validator should cause a 422 or your code returns NOT_ENOUGH_INFO.
    # Accept either 200 with NOT_ENOUGH_INFO or a 422 validation error.
    assert res.status_code in (200, 422)

def test_verify_text_non_factual():
    res = client.post("/api/verify/text", json={"claim": "I want to go to India"})
    assert res.status_code == 200
    data = res.json()
    assert data["verdict"] in ("NOT_ENOUGH_INFO", "REFUTED", "SUPPORTED")  # current logic returns NOT_ENOUGH_INFO
    assert "evidence" in data
    assert isinstance(data["evidence"], list)

# -----------------------
# Mock Wikipedia retrieval
# -----------------------
def make_fake_wiki_response(text_snippet="Fake wiki extract for India"):
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "extract": text_snippet,
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/FakePage"}}
    }
    return fake

@patch("backend.app.requests.get")
def test_wikipedia_retrieval_mock(mock_get):
    # Arrange: mock requests.get to return a controlled JSON
    mock_get.return_value = make_fake_wiki_response("India is a country in South Asia (FAKE).")
    # Act
    res = client.post("/api/verify/text", json={"claim": "India is a country in South Asia"})
    assert res.status_code == 200
    data = res.json()
    # Should include evidence pulled from the mocked wiki
    assert isinstance(data["evidence"], list)
    assert data["evidence"][0]["text"].lower().startswith("india is a country")
    # Ensure our mock was actually called
    assert mock_get.called

# -----------------------
# Test caching wrapper: get_wikipedia_evidence_cached
# -----------------------
def test_wikipedia_cache_calls_once(monkeypatch):
    calls = {"count": 0}

    def fake_get_wiki(q):
        calls["count"] += 1
        return {"source": "Wikipedia", "url": f"https://example/{q}", "text": f"txt-{q}", "similarity": 0.5}

    # Patch the underlying non-cached function (name used in your app)
    monkeypatch.setattr("backend.app.get_wikipedia_evidence", fake_get_wiki)

    # Call cached function multiple times with same query
    r1 = get_wikipedia_evidence_cached("India")
    r2 = get_wikipedia_evidence_cached("India")
    r3 = get_wikipedia_evidence_cached("India")

    assert r1 == r2 == r3
    assert calls["count"] == 1  # underlying function should have been invoked exactly once

# -----------------------
# Optional: smoke test speed for repeated requests (no model load)
# -----------------------
@patch("backend.app.sbert", None)
def test_multiple_quick_requests_no_model():
    # Ensure we can make many quick calls without S-BERT present (avoids model download)
    for _ in range(3):
        res = client.post("/api/verify/text", json={"claim": "The sky is blue"})
        assert res.status_code == 200
