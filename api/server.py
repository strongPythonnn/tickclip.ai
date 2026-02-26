"""
TickClip.ai — FastAPI backend serving fiduciary product evaluations.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.utils import (
    compute_decision,
    enrich_from_amazon,
    fetch_alternatives,
    fetch_diy_articles,
    fetch_keepa,
    search_keepa,
)

load_dotenv()

app = FastAPI(title="TickClip.ai", version="1.0.0")

# CORS — allow the frontend origin in production; permissive for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static frontend — served from /web
# ---------------------------------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")


def _extract_asin(text: str) -> str | None:
    """Try to pull an ASIN from raw input (plain ASIN or Amazon URL)."""
    text = text.strip()
    if ASIN_RE.match(text):
        return text
    # Amazon URL patterns
    m = re.search(r"/(?:dp|gp/product|ASIN)/([A-Z0-9]{10})", text)
    if m:
        return m.group(1)
    return None


@app.get("/api/search")
async def search(q: str = Query(..., min_length=1)):
    """Search for products by keyword.  Returns list of ASINs."""
    keepa_key = os.getenv("KEEPA_API_KEY")
    if not keepa_key:
        raise HTTPException(503, "Keepa API key is not configured.")

    # If the query looks like an ASIN or Amazon URL, short-circuit
    asin = _extract_asin(q)
    if asin:
        return {"results": [{"asin": asin, "title": None, "image": None}]}

    try:
        results = await search_keepa(q, keepa_key)
    except Exception as exc:
        raise HTTPException(502, f"Keepa search failed: {exc}")

    if not results:
        raise HTTPException(404, "No products found.")

    return {"results": results}


@app.get("/api/evaluate")
async def evaluate(asin: str = Query(..., min_length=10, max_length=10)):
    """Full fiduciary evaluation for a single ASIN."""
    if not ASIN_RE.match(asin):
        raise HTTPException(400, "Invalid ASIN format.")

    keepa_key = os.getenv("KEEPA_API_KEY")
    if not keepa_key:
        raise HTTPException(503, "Keepa API key is not configured.")

    # 1. Keepa data
    try:
        keepa = await fetch_keepa(asin, keepa_key)
    except Exception as exc:
        raise HTTPException(502, f"Keepa request failed: {exc}")

    # 2. Optional Amazon PA-API enrichment
    amazon = await enrich_from_amazon(asin)
    if amazon:
        if amazon.get("title"):
            keepa["title"] = amazon["title"]
        if amazon.get("image"):
            keepa["image"] = amazon["image"]

    # 3. Decision engine
    decision_result = compute_decision(
        keepa["current_price"],
        keepa["hist_low"],
        keepa["avg_90d"],
        keepa["volatility_score"],
        keepa["seller_risk"],
    )

    # 4. Alternatives + DIY if CLIP or SKIP
    alternatives: list[dict] = []
    diy_articles: list[dict] = []
    if decision_result["decision"] in ("CLIP", "SKIP"):
        try:
            alternatives = await fetch_alternatives(keepa["title"])
        except Exception:
            pass
        try:
            diy_articles = await fetch_diy_articles(keepa["title"])
        except Exception:
            pass

    return {
        "asin": asin,
        "title": keepa["title"],
        "image": keepa["image"],
        "current_price": keepa["current_price"],
        "currency": keepa["currency"],
        "hist_low": keepa["hist_low"],
        "avg_90d": keepa["avg_90d"],
        "volatility_score": keepa["volatility_score"],
        "seller_risk": keepa["seller_risk"],
        "decision": decision_result["decision"],
        "confidence": decision_result["confidence"],
        "explanation": decision_result["explanation"],
        "price_series": keepa["price_series"],
        "alternatives": alternatives,
        "diy_articles": diy_articles,
    }


# ---------------------------------------------------------------------------
# Serve frontend — must be AFTER API routes
# ---------------------------------------------------------------------------

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve index.html for any non-API route (SPA fallback)."""
        file = WEB_DIR / full_path
        if file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(WEB_DIR / "index.html"))
