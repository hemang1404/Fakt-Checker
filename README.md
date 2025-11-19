# Fakt-Checker

A minimal MVP: a Chrome extension + FastAPI backend to quickly verify text claims and detect manipulated media. This repo contains the backend scaffold (FastAPI) and extension skeleton (Chrome Extension Manifest v3 + popup).

> NOTE: This is a development demo. Do **not** use `allow_origins=["*"]` in production. See `SECURITY.md` for guidance.

---

## Quick demo (run locally)

### Prerequisites
- Python 3.10+ (3.11 recommended)
- Chrome (for loading the extension)
- git

### Backend (local)
```bash
# from repo root
cd backend

# create venv (Linux/macOS)
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
# python -m venv venv
# .\venv\Scripts\Activate.ps1

pip install -r requirements.txt

# run dev server
uvicorn app:app --reload --host 127.0.0.1 --port 8000
