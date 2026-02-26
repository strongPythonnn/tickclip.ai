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
    batch_evaluate_asins,
    compute_decision,
    enrich_from_amazon,
    fetch_alternatives,
    fetch_deals,
    fetch_diy_articles,
    fetch_keepa,
    fetch_retailer_prices,
    fetch_reviews,
    search_amazon_deals,
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
    """Search for products by keyword.  Returns list of ASINs with TICK/CLIP/SKIP decisions."""
    keepa_key = os.getenv("KEEPA_API_KEY")
    if not keepa_key:
        raise HTTPException(503, "Keepa API key is not configured.")

    # If the query looks like an ASIN or Amazon URL, short-circuit
    asin = _extract_asin(q)
    if asin:
        # Evaluate single ASIN immediately
        evals = await batch_evaluate_asins([asin], keepa_key)
        info = evals.get(asin, {})
        return {"results": [{
            "asin": asin,
            "title": info.get("title"),
            "image": info.get("image"),
            "current_price": info.get("current_price"),
            "decision": info.get("decision", "CLIP"),
            "confidence": info.get("confidence", 0),
        }]}

    try:
        results = await search_keepa(q, keepa_key)
    except Exception:
        results = []

    if not results:
        return {"results": []}

    # Batch evaluate all found ASINs to get TICK/CLIP/SKIP
    asins = [r["asin"] for r in results]
    evals = await batch_evaluate_asins(asins, keepa_key)

    # Merge evaluation data into results
    enriched = []
    for r in results:
        info = evals.get(r["asin"], {})
        enriched.append({
            "asin": r["asin"],
            "title": info.get("title") or r.get("title"),
            "image": info.get("image") or r.get("image"),
            "current_price": info.get("current_price"),
            "decision": info.get("decision", "CLIP"),
            "confidence": info.get("confidence", 0),
        })

    return {"results": enriched}


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
    amazon_offers: list[dict] = []
    amazon_promotions: list[dict] = []
    amazon_offer_summaries: list[dict] = []
    amazon_features: list[str] = []
    brand = None
    if amazon:
        if amazon.get("title"):
            keepa["title"] = amazon["title"]
        if amazon.get("image"):
            keepa["image"] = amazon["image"]
        amazon_offers = amazon.get("offers", [])
        amazon_promotions = amazon.get("promotions", [])
        amazon_offer_summaries = amazon.get("offer_summaries", [])
        amazon_features = amazon.get("features", [])
        brand = amazon.get("brand") or amazon.get("manufacturer")

    # 3. Decision engine (includes manipulation analysis)
    decision_result = compute_decision(
        keepa["current_price"],
        keepa["hist_low"],
        keepa["avg_90d"],
        keepa["volatility_score"],
        keepa["seller_risk"],
        keepa.get("price_manipulation"),
    )

    # 4. Fetch everything in parallel — always fetch all sections
    import asyncio

    results = await asyncio.gather(
        fetch_retailer_prices(keepa["title"]),
        fetch_deals(keepa["title"]),
        fetch_reviews(keepa["title"]),
        fetch_alternatives(keepa["title"]),
        fetch_diy_articles(keepa["title"]),
        search_amazon_deals(keepa["title"]),
        return_exceptions=True,
    )

    retailer_prices = results[0] if isinstance(results[0], list) else []
    deals = results[1] if isinstance(results[1], list) else []
    review_data = results[2] if isinstance(results[2], dict) else {"reviews": [], "summary": {}}
    alternatives = results[3] if isinstance(results[3], list) else []
    diy_articles = results[4] if isinstance(results[4], list) else []
    amazon_deals = results[5] if isinstance(results[5], list) else []

    return {
        "asin": asin,
        "title": keepa["title"],
        "image": keepa["image"],
        "brand": brand,
        "features": amazon_features,
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
        "price_manipulation": keepa.get("price_manipulation", {}),
        "amazon_offers": amazon_offers,
        "amazon_promotions": amazon_promotions,
        "amazon_offer_summaries": amazon_offer_summaries,
        "amazon_deals": amazon_deals,
        "retailer_prices": retailer_prices,
        "deals": deals,
        "review_analysis": review_data,
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
