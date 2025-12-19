from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from fastapi.middleware.cors import CORSMiddleware
import requests
import re
import logging
import os
from urllib.parse import quote_plus
from time import perf_counter
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger= logging.getLogger(__name__)

# API Keys (optional - set as environment variable)
GOOGLE_FACT_CHECK_API_KEY = os.environ.get("GOOGLE_FACT_CHECK_API_KEY")
GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Sentence transformers for similarity (lazy loading)
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

# Optional: eager-load SBERT at startup to avoid first-call delay.
# WARNING: this increases startup time and memory usage.
# Disabled to avoid model registry conflicts
sbert = None


# Optional spaCy entity extraction
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    nlp = None

# Optional Groq LLM for claim analysis
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    groq_client = None


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
    """Fetch evidence from Wikipedia API."""
    if not query:
        return {"source": "Wikipedia", "url": None, "text": "No relevant evidence found", "similarity": 0.0}

    try:
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
                "source": "Wikipedia",
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", search_url),
                "text": data.get("extract", "No extract found"),
                "similarity": 0.0,  # Will be calculated later
            }
    except Exception as e:
        logger.warning(f"Wikipedia API error: {e}")

    return {"source": "Wikipedia", "url": None, "text": "No relevant evidence found", "similarity": 0.0}


def get_dbpedia_evidence(query: str):
    """Fetch evidence from DBpedia SPARQL endpoint."""
    if not query:
        return {"source": "DBpedia", "url": None, "text": "No relevant evidence found", "similarity": 0.0}
    
    # DBpedia SPARQL endpoint
    sparql_url = "http://dbpedia.org/sparql"
    
    # Simple query to get abstract about the entity
    sparql_query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbr: <http://dbpedia.org/resource/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?abstract ?resource WHERE {{
        ?resource rdfs:label ?label .
        ?resource dbo:abstract ?abstract .
        FILTER (lang(?abstract) = 'en')
        FILTER (regex(?label, "^{query}$", "i"))
    }} LIMIT 1
    """
    
    try:
        response = requests.get(
            sparql_url,
            params={"query": sparql_query, "format": "json"},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", {}).get("bindings", [])
            
            if results:
                abstract = results[0].get("abstract", {}).get("value", "")
                resource_url = results[0].get("resource", {}).get("value", "")
                
                # Limit abstract length
                if len(abstract) > 500:
                    abstract = abstract[:500] + "..."
                
                return {
                    "source": "DBpedia",
                    "url": resource_url,
                    "text": abstract,
                    "similarity": 0.0,
                }
    except Exception as e:
        logger.warning(f"DBpedia API error: {e}")
    
    return {"source": "DBpedia", "url": None, "text": "No relevant evidence found", "similarity": 0.0}


def get_wikidata_evidence(query: str):
    """Fetch evidence from Wikidata."""
    if not query:
        return {"source": "Wikidata", "url": None, "text": "No relevant evidence found", "similarity": 0.0}
    
    try:
        # Search for entity
        search_url = "https://www.wikidata.org/w/api.php"
        search_params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
            "limit": 1
        }
        
        response = requests.get(search_url, params=search_params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("search", [])
            
            if results:
                entity = results[0]
                entity_id = entity.get("id")
                label = entity.get("label", "")
                description = entity.get("description", "")
                
                # Get more details about the entity
                if entity_id:
                    entity_url = f"https://www.wikidata.org/wiki/{entity_id}"
                    text = f"{label}: {description}" if description else label
                    
                    return {
                        "source": "Wikidata",
                        "url": entity_url,
                        "text": text,
                        "similarity": 0.0,
                    }
    except Exception as e:
        logger.warning(f"Wikidata API error: {e}")
    
    return {"source": "Wikidata", "url": None, "text": "No relevant evidence found", "similarity": 0.0}


def get_google_factcheck_evidence(claim: str):
    """Fetch evidence from Google Fact Check API."""
    if not claim or not GOOGLE_FACT_CHECK_API_KEY:
        if not GOOGLE_FACT_CHECK_API_KEY:
            logger.info("Google Fact Check API key not configured")
        return {"source": "Google Fact Check", "url": None, "text": "API key not configured", "similarity": 0.0}
    
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": claim,
            "key": GOOGLE_FACT_CHECK_API_KEY,
            "languageCode": "en"
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", [])
            
            if claims:
                # Get the first claim check
                claim_review = claims[0]
                text_parts = []
                
                # Get claim text
                claim_text = claim_review.get("text", "")
                if claim_text:
                    text_parts.append(f"Claim: {claim_text}")
                
                # Get review verdict
                reviews = claim_review.get("claimReview", [])
                if reviews:
                    review = reviews[0]
                    rating = review.get("textualRating", "")
                    publisher = review.get("publisher", {}).get("name", "")
                    review_url = review.get("url", "")
                    
                    if rating:
                        text_parts.append(f"Rating: {rating}")
                    if publisher:
                        text_parts.append(f"by {publisher}")
                    
                    return {
                        "source": "Google Fact Check",
                        "url": review_url or None,
                        "text": " | ".join(text_parts) if text_parts else "Fact check found",
                        "similarity": 0.0,
                    }
        
        elif response.status_code == 429:
            logger.warning("Google Fact Check API rate limit exceeded")
        
    except Exception as e:
        logger.warning(f"Google Fact Check API error: {e}")
    
    return {"source": "Google Fact Check", "url": None, "text": "No relevant evidence found", "similarity": 0.0}


def get_google_search_evidence(query: str):
    """Fetch evidence from Google Custom Search API."""
    if not query or not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
            logger.info("Google Search API key or Search Engine ID not configured")
        return {"source": "Google Search", "url": None, "text": "API key not configured", "similarity": 0.0}
    
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_SEARCH_API_KEY,
            "cx": GOOGLE_SEARCH_ENGINE_ID,
            "q": query,
            "num": 3  # Get top 3 results
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            
            if items:
                # Combine snippets from top results
                snippets = []
                urls = []
                
                for item in items[:3]:
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")
                    title = item.get("title", "")
                    
                    if snippet:
                        snippets.append(f"{title}: {snippet}")
                    if link:
                        urls.append(link)
                
                combined_text = " | ".join(snippets) if snippets else "Results found"
                primary_url = urls[0] if urls else None
                
                return {
                    "source": "Google Search",
                    "url": primary_url,
                    "text": combined_text[:500],  # Limit length
                    "similarity": 0.0,
                }
        
        elif response.status_code == 429:
            logger.warning("Google Search API rate limit exceeded")
        
    except Exception as e:
        logger.warning(f"Google Search API error: {e}")
    
    return {"source": "Google Search", "url": None, "text": "No relevant evidence found", "similarity": 0.0}


def get_multiple_evidence_sources(query: str, full_claim: str = ""):
    """Fetch evidence from multiple sources."""
    sources = []
    
    # Wikipedia (primary source)
    wiki = get_wikipedia_evidence(query)
    if wiki["text"] and wiki["text"] != "No relevant evidence found":
        sources.append(wiki)
    
    # DBpedia (structured data)
    dbpedia = get_dbpedia_evidence(query)
    if dbpedia["text"] and dbpedia["text"] != "No relevant evidence found":
        sources.append(dbpedia)
    
    # Wikidata (knowledge base)
    wikidata = get_wikidata_evidence(query)
    if wikidata["text"] and wikidata["text"] != "No relevant evidence found":
        sources.append(wikidata)
    
    # Google Search (web search results)
    if GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
        google_search = get_google_search_evidence(full_claim or query)
        if google_search["text"] and google_search["text"] not in ["No relevant evidence found", "API key not configured"]:
            sources.append(google_search)
    
    # Google Fact Check (fact-checking specific)
    if full_claim and GOOGLE_FACT_CHECK_API_KEY:
        google_fc = get_google_factcheck_evidence(full_claim)
        if google_fc["text"] and google_fc["text"] not in ["No relevant evidence found", "API key not configured"]:
            sources.append(google_fc)
    
    return sources


def calculate_evidence_verdict(similarities: list[float]) -> tuple[str, float]:
    """
    Determine verdict based on multiple evidence similarity scores.
    Returns: (verdict, confidence_score)
    """
    if not similarities:
        return "NOT_ENOUGH_INFO", 0.0
    
    avg_similarity = sum(similarities) / len(similarities)
    
    # Calculate confidence based on agreement between sources
    if len(similarities) > 1:
        # Standard deviation - lower means more agreement
        variance = sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities)
        std_dev = variance ** 0.5
        # Higher agreement = higher confidence
        agreement_factor = max(0.0, 1.0 - std_dev)
    else:
        # Single source - confidence based on similarity strength
        agreement_factor = 0.8
    
    # Determine verdict based on average similarity
    if avg_similarity > 0.65:
        verdict = "SUPPORTED"
        confidence = avg_similarity * agreement_factor
    elif avg_similarity < 0.35:
        verdict = "REFUTED"
        confidence = (1.0 - avg_similarity) * agreement_factor
    else:
        verdict = "NOT_ENOUGH_INFO"
        confidence = 0.5 * agreement_factor
    
    return verdict, round(confidence, 3)


# Small in-memory cache to avoid repeated Wikipedia calls for the same query
@lru_cache(maxsize=256)
def get_wikipedia_evidence_cached(query: str):
    # lru_cache requires hashable args and works well for short query strings
    return get_wikipedia_evidence(query)


def evidence_similarity(claim: str, evidence_text: str) -> float:
    global sbert
    
    # Lazy load the model on first use
    if sbert is None and SentenceTransformer is not None:
        try:
            sbert = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            return 0.0
    
    if sbert is None:
        return 0.0
        
    try:
        a= sbert.encode(claim, convert_to_tensor=True)
        b= sbert.encode(evidence_text, convert_to_tensor=True)
        sim= float(util.cos_sim(a, b).item())
        sim = max(0.0, min(1.0, (sim + 1)/2))
        return sim
    except Exception:
        return 0.0

def is_factual(text: str) -> bool:
    """
    Enhanced claim detector using LLM (Groq) if available, with rule-based fallback.
    Returns True when the text looks like a factual assertion we can check.
    """
    if not text:
        return False

    # Try LLM-powered detection first
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fact-checking assistant. Analyze if the given text contains a verifiable factual claim (not an opinion, question, or personal statement). Respond with only 'YES' or 'NO'."
                    },
                    {
                        "role": "user",
                        "content": f"Does this contain a verifiable factual claim?\n\n\"{text}\""
                    }
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().upper()
            logger.info(f"LLM claim analysis: '{text[:50]}...' -> {answer}")
            return "YES" in answer
        except Exception as e:
            logger.warning(f"Groq LLM error, falling back to rules: {e}")
    
    # Fallback: Rule-based detector
    c = text.strip().lower()

    # Ends with a question mark -> not factual
    if c.endswith("?"):
        return False

    # Personal / intention patterns -> not factual
    personal_prefixes = (
        r"^i want", r"^i will", r"^i'm going", r"^i am going", r"^i plan",
        r"^can i", r"^should i", r"^do i", r"^how do i", r"^how can i",
        r"^i have to", r"^i'd like", r"^i would like"
    )
    for pat in personal_prefixes:
        if re.match(pat, c):
            return False

    # Question-style words -> not factual
    if re.match(r'^(how|why|where|when|who|what|does|do|did)\b', c):
        return False

    # Opinion words -> not factual
    if re.search(r"\b(i think|i believe|i feel|maybe|probably|possibly|might be)\b", c):
        return False

    # Factual verbs / structures -> likely factual
    if re.search(r'\b(is|are|was|were|has|have|had|consists|causes|caused|leads|led|results)\b', c):
        return True

    # Conservative default -> not factual
    return False




class ClaimReq(BaseModel):
    claim: str
    
    @field_validator('claim')
    @classmethod
    def validate_claim(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Claim cannot be empty')
        if len(v) > 2000:
            raise ValueError('Claim too long (max 2000 characters)')
        return v


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
    start = perf_counter()

    if not claim:
        return {"verdict": "NOT_ENOUGH_INFO", "score": 0.0,
                "evidence": [], "explanation": "Empty claim."}

    lower = claim.lower()
    positive_words = ["definitely", "always", "confirmed", "proven", "true"]
    negative_words = ["fake", "hoax", "lie", "false", "not", "never"]
    doubt_words = ["maybe", "unclear", "rumor", "rumour", "suspected", "probably"]

    # Use word boundaries to avoid false matches (e.g., "not" in "note")
    pos_hits = sum(1 for w in positive_words if re.search(rf'\b{re.escape(w)}\b', lower))
    neg_hits = sum(1 for w in negative_words if re.search(rf'\b{re.escape(w)}\b', lower))
    doubt_hits = sum(1 for w in doubt_words if re.search(rf'\b{re.escape(w)}\b', lower))

    score = 0.5 + 0.2 * (pos_hits - neg_hits)
    score = max(0.0, min(1.0, score))

    if neg_hits > pos_hits:
        verdict = "REFUTED"
    elif pos_hits > neg_hits:
        verdict = "SUPPORTED"
    else:
        verdict = "NOT_ENOUGH_INFO"

    # determine if claim is factual
    factual = is_factual(claim)
    logger.info(f"Claim: '{claim[:60]}' | factual={factual} | pos={pos_hits} neg={neg_hits}")

    # prepare evidence only for factual claims
    evidence = []
    confidence = 0.0

    if factual:
        keyword = extract_entity(claim)
        logger.info(f"Extracted keyword: '{keyword}'")
        
        # Fetch evidence from multiple sources (pass full claim for Google Fact Check)
        evidence_sources = get_multiple_evidence_sources(keyword, full_claim=claim)
        
        if evidence_sources:
            # Calculate similarity for each evidence source
            similarities = []
            for src in evidence_sources:
                if src["text"] and src["text"] != "No relevant evidence found":
                    sim = evidence_similarity(claim, src["text"])
                    src["similarity"] = round(sim, 3)
                    similarities.append(sim)
                    logger.info(f"Source: {src['source']} | Similarity: {sim:.3f}")
            
            # Determine verdict based on all evidence
            if similarities:
                verdict, confidence = calculate_evidence_verdict(similarities)
                evidence = evidence_sources
            else:
                # No valid evidence found
                verdict = "NOT_ENOUGH_INFO"
                confidence = 0.0
                evidence = []
        else:
            # No evidence sources returned
            verdict = "NOT_ENOUGH_INFO"
            confidence = 0.0
            evidence = []
        
        explanation = (
            f"Analyzed factual claim: '{claim[:150]}' | "
            f"Keyword: '{keyword}' | "
            f"Evidence sources: {len(evidence)} | "
            f"Verdict: {verdict} | "
            f"Confidence: {confidence}"
        )
        
    else:
        # Non-factual: return heuristic result without evidence
        # Use heuristic for confidence on non-factual claims
        confidence = round(abs(score - 0.5) * 2, 3)  # 0 = uncertain, 1 = very certain
        
        explanation = (
            f"Non-factual or personal/question claim detected. "
            f"Heuristic analysis: positive={pos_hits}, negative={neg_hits}, doubt={doubt_hits}. "
            f"No evidence sources consulted."
        )
        logger.info("Non-factual path: %s", explanation)

    elapsed = perf_counter() - start
    logger.info("Processed request in %.3fs | factual=%s | verdict=%s | confidence=%.3f", 
                elapsed, factual, verdict, confidence)
    
    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence,
        "explanation": explanation,
        "metadata": {
            "factual": factual,
            "processing_time_ms": round(elapsed * 1000, 2),
            "sources_consulted": len(evidence)
        }
    }

