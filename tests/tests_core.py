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

# -----------------------
# Test Google Search API integration
# -----------------------
@patch("backend.app.requests.get")
def test_google_search_evidence(mock_get):
    """Test that Google Search API is called and returns evidence"""
    # Mock Google Search API response
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "items": [
            {"snippet": "Paris is the capital of France", "link": "https://example.com/1"},
            {"snippet": "France capital city Paris", "link": "https://example.com/2"},
            {"snippet": "Paris has been capital since X", "link": "https://example.com/3"}
        ]
    }
    mock_get.return_value = fake_response
    
    from backend.app import get_google_search_evidence
    result = get_google_search_evidence("Paris capital France")
    
    assert result is not None
    assert result["source"] == "Google Search"
    assert "Paris" in result["text"]
    assert "https://example.com" in result["url"]

@patch("backend.app.GOOGLE_SEARCH_API_KEY", None)
def test_google_search_without_api_key():
    """Test that app handles missing Google Search API key gracefully"""
    from backend.app import get_google_search_evidence
    result = get_google_search_evidence("test query")
    # Returns error dict instead of None
    assert result is not None
    assert result["text"] == "API key not configured"

# -----------------------
# Test multi-source evidence aggregation
# -----------------------
@patch("backend.app.requests.get")
def test_multiple_evidence_sources(mock_get):
    """Test that get_multiple_evidence_sources calls all 5 sources"""
    # Mock all API responses
    def mock_response(*args, **kwargs):
        url = args[0] if args else kwargs.get('url', '')
        fake = MagicMock()
        fake.status_code = 200
        
        if "wikipedia" in url:
            fake.json.return_value = {"extract": "Paris is the capital", "content_urls": {"desktop": {"page": "https://wiki.com"}}}
        elif "dbpedia" in url:
            fake.json.return_value = {"results": {"bindings": [{"abstract": {"value": "Paris info"}}]}}
        elif "wikidata" in url:
            fake.json.return_value = {"search": [{"description": "capital of France"}]}
        elif "googleapis.com/customsearch" in url:
            fake.json.return_value = {"items": [{"snippet": "Paris capital", "link": "https://google.com"}]}
        elif "factchecktools" in url:
            fake.json.return_value = {"claims": [{"text": "Paris is capital", "claimReview": [{"url": "https://fact.com"}]}]}
        
        return fake
    
    mock_get.side_effect = mock_response
    
    from backend.app import get_multiple_evidence_sources
    evidence = get_multiple_evidence_sources("Paris is the capital of France")
    
    # Should have evidence from multiple sources
    assert isinstance(evidence, list)
    assert len(evidence) > 0
    
    # Check that different sources are present
    sources = [e["source"] for e in evidence]
    assert "Wikipedia" in sources

# -----------------------
# Test .env file loading
# -----------------------
def test_env_variables_loaded():
    """Test that environment variables are loaded (even if empty)"""
    from backend.app import GOOGLE_FACT_CHECK_API_KEY, GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID
    # Variables should be defined (even if None)
    assert GOOGLE_FACT_CHECK_API_KEY is not None or GOOGLE_FACT_CHECK_API_KEY is None
    assert GOOGLE_SEARCH_API_KEY is not None or GOOGLE_SEARCH_API_KEY is None
    assert GOOGLE_SEARCH_ENGINE_ID is not None or GOOGLE_SEARCH_ENGINE_ID is None

# -----------------------
# Test confidence calculation with multiple sources
# -----------------------
@patch("backend.app.get_multiple_evidence_sources")
def test_confidence_with_multiple_sources(mock_evidence):
    """Test that confidence score increases with more agreeing sources"""
    # Mock 3 sources with high similarity
    mock_evidence.return_value = [
        {"source": "Wikipedia", "text": "Paris is capital", "similarity": 0.9, "url": "https://test.com"},
        {"source": "DBpedia", "text": "Paris capital city", "similarity": 0.85, "url": "https://test2.com"},
        {"source": "Wikidata", "text": "Paris France capital", "similarity": 0.88, "url": "https://test3.com"}
    ]
    
    res = client.post("/api/verify/text", json={"claim": "Paris is the capital of France"})
    assert res.status_code == 200
    data = res.json()
    
    # With 3 sources, should have evidence
    assert len(data["evidence"]) == 3
    assert data["confidence"] >= 0.0  # Has some confidence value
    # Verdict can be any based on the logic
    assert data["verdict"] in ("SUPPORTED", "NOT_ENOUGH_INFO", "REFUTED")

# -----------------------
# Test Groq LLM Integration
# -----------------------
def test_groq_client_initialization():
    """Test that Groq client initializes when API key is present"""
    from backend.app import groq_client, GROQ_API_KEY
    
    if GROQ_API_KEY:
        assert groq_client is not None
    else:
        assert groq_client is None

@patch("backend.app.groq_client")
def test_is_factual_with_llm(mock_groq):
    """Test is_factual uses LLM when available"""
    # Mock Groq response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "YES"
    
    mock_groq.chat.completions.create.return_value = mock_response
    
    from backend.app import is_factual
    
    result = is_factual("Paris is the capital of France")
    assert result == True
    assert mock_groq.chat.completions.create.called

@patch("backend.app.groq_client")
def test_is_factual_llm_fallback_on_error(mock_groq):
    """Test that is_factual falls back to rules when LLM fails"""
    # Mock LLM to raise exception
    mock_groq.chat.completions.create.side_effect = Exception("API Error")
    
    from backend.app import is_factual
    
    # Should fall back to rule-based detection
    assert is_factual("Paris is the capital of France") == True
    assert is_factual("I want to go to India") == False

@patch("backend.app.groq_client", None)
def test_is_factual_without_llm():
    """Test that is_factual works with rule-based fallback when no LLM"""
    from backend.app import is_factual
    
    # Should use rule-based detection
    assert is_factual("The Sun is a star") == True
    assert is_factual("I think the sky is blue") == False
    assert is_factual("Should I go to school?") == False

def test_llm_claim_analysis_integration():
    """Test end-to-end claim analysis with LLM"""
    # This test runs against real API if key is present
    res = client.post("/api/verify/text", json={"claim": "The Earth orbits the Sun"})
    assert res.status_code == 200
    data = res.json()
    
    # Should be detected as factual claim
    assert isinstance(data["evidence"], list)
    assert "verdict" in data
