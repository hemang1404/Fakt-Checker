from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

def get_wikipedia_evidence(query: str):
    search_url=f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    headers={
        "User-Agent": "Fakt-Checker/1.0 (to.be.or.what.to.be@gmail.com)"}
    try:
        response = requests.get(search_url, headers=headers, timeout=3)
        if response.status_code == 200:
            data = response.json()
            return {
                "search_url": data.get("content_urls", {}).get("desktop", {}).get("page", search_url),
                "text": data.get("extract", "No extract found"),
                "score": 0.6
            }
    except Exception:
        pass

    return {
        "search_url": None,
        "text": "No relevant evidence found",
        "score": 0.0
    }

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
    claim = req.claim.strip()
    if not claim:
        return {"verdict": "NOT_ENOUGH_INFO", "score": 0.0, "evidence": [], "explanation": "Empty claim."}

    # Smarter heuristic: look for positive/negative/doubt words
    lower = claim.lower()

    positive_words = ["definitely", "always", "confirmed", "proven", "true"]
    negative_words = ["fake", "hoax", "lie", "false", "not", "never"]
    doubt_words = ["maybe", "unclear", "rumor", "rumour", "suspected", "probably"]

    pos_hits = sum(word in lower for word in positive_words)
    neg_hits = sum(word in lower for word in negative_words)
    doubt_hits = sum(word in lower for word in doubt_words)

    score = 0.5 + 0.2 * (pos_hits - neg_hits)
    score = max(0.0, min(1.0, score))

    if neg_hits > pos_hits:
        verdict = "REFUTED"
    elif pos_hits > neg_hits:
        verdict = "SUPPORTED"
    else:
        verdict = "NOT_ENOUGH_INFO"

    keyword = claim.split()[0]
    evidence = [get_wikipedia_evidence(keyword)]
    return {
        "verdict": verdict,
        "score": score,
        "evidence": evidence,
        "explanation": (
            f"Simple heuristic verdict for claim: {claim[:200]} "
            f"(pos_hits={pos_hits}, neg_hits={neg_hits}, doubt_hits={doubt_hits})"
        )
    }

