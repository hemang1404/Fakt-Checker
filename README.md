# 🔍 Fakt-Checker

An AI-powered fact-checking system that verifies claims using multiple evidence sources and LLM-enhanced analysis.

## ✨ Features

- **Multi-Source Evidence Aggregation**: Queries 5+ authoritative sources
  - Wikipedia REST API
  - DBpedia (structured knowledge)
  - Wikidata (entity descriptions)
  - Google Fact Check Tools
  - Google Custom Search
  
- **LLM-Powered Analysis**: Uses Groq (Llama-3.3-70B) for intelligent claim detection
  - Distinguishes facts from opinions with contextual understanding
  - Automatic fallback to rule-based detection
  
- **Smart Confidence Scoring**: Evidence-based verdicts with multi-source agreement
  
- **Semantic Similarity**: SBERT embeddings for accurate claim-evidence matching

## 🛠️ Tech Stack

**Backend:**
- FastAPI (async Python web framework)
- Sentence Transformers (semantic similarity)
- spaCy (entity extraction)
- Groq API (LLM integration)
- Pydantic v2 (data validation)

**Frontend:**
- Vanilla HTML/CSS/JavaScript
- Responsive design with gradient theme

**Testing:**
- pytest with 20+ comprehensive test cases
- Mock API calls for reliability

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/hemang1404/Fakt-Checker.git
cd Fakt-Checker
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r backend/requirements.txt
```

4. **Set up environment variables (optional but recommended):**
```bash
# Copy example file
cp backend/.env.example backend/.env

# Add your API keys in backend/.env:
# - GROQ_API_KEY (free at console.groq.com)
# - GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_ENGINE_ID
# - GOOGLE_FACT_CHECK_API_KEY
```

5. **Run the backend:**
```bash
uvicorn backend.app:app --reload --port 8000
```

6. **Open the frontend:**
```
Open frontend/index.html in your browser
```

## 📡 API Documentation

### Verify Text Endpoint

**POST** `/api/verify/text`

**Request:**
```json
{
  "claim": "Paris is the capital of France"
}
```

**Response:**
```json
{
  "verdict": "SUPPORTED",
  "confidence": 0.85,
  "evidence": [
    {
      "source": "Wikipedia",
      "text": "Paris is the capital and most populous city...",
      "url": "https://en.wikipedia.org/wiki/Paris",
      "similarity": 0.92
    }
  ],
  "explanation": "High confidence based on 3 agreeing sources",
  "metadata": {
    "processing_time_ms": 1234,
    "sources_queried": 5,
    "llm_used": true
  }
}
```

## 🧪 Running Tests

```bash
pytest tests/ -v
# or with venv
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 20 tests should pass.

## 📊 Project Status

- ✅ Multi-source evidence aggregation (5 sources)
- ✅ LLM-powered claim analysis (Groq)
- ✅ Frontend demo interface
- ✅ Comprehensive test suite (20 tests)
- 🚧 Deployment (coming soon)
- 🚧 Chrome browser extension (planned)
- 🚧 Database for claim history (planned)

## 🔮 Future Plans

1. **Chrome Extension**: Right-click selected text → instant fact-check
2. **Claim Database**: Track popular claims and historical verdicts
3. **Production Deployment**: Host on Render/Railway
4. **Enhanced UI**: React/Vue frontend with better UX
5. **API Rate Limiting**: Production-ready with authentication

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📄 License

MIT License - feel free to use for learning or portfolio purposes.

## 👤 Author

**Hemang** - [GitHub](https://github.com/hemang1404)

---

*Built to demonstrate full-stack development, API integration, and AI/ML skills.*
