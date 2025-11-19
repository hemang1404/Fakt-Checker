from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class ClaimReq(BaseModel):
    claim: str


app = FastAPI(title="Fakt-Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "name": "Fakt-Checker"}


@app.post("/api/verify/text")
def verify_text(req: ClaimReq):
    claim = req.claim
    if not claim:
        return {"verdict": "NOT_ENOUGH_INFO", "score": 0.0, "evidence": [], "explanation": "Empty claim."}

    # Dummy heuristic — change this to your own rule to test
    lower = claim.lower()
    if "not" in lower or "never" in lower:
        verdict = "REFUTED"
        score = 0.12
    elif "always" in lower or "definitely" in lower:
        verdict = "SUPPORTED"
        score = 0.88
    else:
        verdict = "NOT_ENOUGH_INFO"
        score = 0.5

    evidence = [
        {"source": "https://example.com/sample", "text": "Example evidence snippet", "score": 0.45}
    ]
    explanation = f"Simple heuristic verdict for claim: {claim[:200]}"
    return {
        "verdict": verdict,
        "score": score,
        "evidence": evidence,
        "explanation": explanation,
    }
