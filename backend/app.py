from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

# Optional spaCy entity extraction
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    nlp = None


def extract_entity_fallback(text: str, max_tokens: int = 3) -> str:
    words = text.replace("?", "").replace(".", "").replace(",", "").split()
    return " ".join(words[:max_tokens]) if words else "fact"


def extract_entity(text: str) -> str:
    if nlp is None:
        return extract_entity_fallback(text)

    try:
        doc = nlp(text)
        ents = [ent.text for ent in doc.ents if ent.label_ in (
            "PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART", "NORP")]
        if ents:
            return max(ents, key=len)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        if noun_chunks:
            return noun_chunks[0]
    except Exception:
        pass

    return extract_entity_fallback(text)


def get_wikipedia_evidence(query: str):
    if not query:
        return {"source": None, "text": "No relevant evidence found", "score": 0.0}

    try:
        from urllib.parse import quote_plus
        q = quote_plus(query)
    except Exception:
        q = query.replace(" ", "%20")

    search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
    headers = {"User-Agent": "Fakt-Checker/1.0 (to.be.or.what.to.be@gmail.com)"}

    try:
        response = requests.get(search_url, headers=headers, timeout=3)
        if response.status_code == 200:
            data = response.json()
            return {
                "source": data.get("content_urls", {}).get("desktop", {}).get("page", search_url),
                "text": data.get("extract", "No extract found"),
                "score": 0.6,
            }
    except Exception:
        pass

    return {"source": None, "text": "No relevant evidence found", "score": 0.0}


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
        return {"verdict": "NOT_ENOUGH_INFO", "score": 0.0,
                "evidence": [], "explanation": "Empty claim."}

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

    keyword = extract_entity(claim)
    wiki = get_wikipedia_evidence(keyword)
    evidence = [wiki]

    explanation = (
        f"Simple heuristic verdict for claim: {claim[:200]} "
        f"(pos_hits={pos_hits}, neg_hits={neg_hits}, "
        f"doubt_hits={doubt_hits}, keyword={keyword})"
    )

    return {
        "verdict": verdict,
        "score": round(score, 3),
        "evidence": evidence,
        "explanation": explanation,
    }
