"""
TickClip.ai — Utility helpers for Keepa, Amazon PA-API, decision engine, and web search.
"""

import os
import math
import time
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlencode

import httpx

# ---------------------------------------------------------------------------
# Keepa helpers
# ---------------------------------------------------------------------------

KEEPA_BASE = "https://api.keepa.com"


def _keepa_time_to_unix(keepa_min: int) -> int:
    """Convert Keepa minute timestamp to Unix epoch seconds."""
    return (keepa_min + 21564000) * 60


def _parse_keepa_csv(csv_data: list[int | None]) -> list[dict]:
    """
    Keepa stores price history as flat pairs [time, price, time, price, …].
    Prices are in cents (or -1 for out-of-stock).  Returns list of {date, price}.
    """
    points: list[dict] = []
    if not csv_data:
        return points
    i = 0
    while i + 1 < len(csv_data):
        t = csv_data[i]
        p = csv_data[i + 1]
        i += 2
        if t is None or p is None or p < 0:
            continue
        unix_ts = _keepa_time_to_unix(t)
        points.append({
            "date": datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d"),
            "price": round(p / 100, 2),
        })
    return points


async def fetch_keepa(asin: str, api_key: str) -> dict[str, Any]:
    """
    Call Keepa product endpoint.  Returns dict with:
      title, image, current_price, currency, hist_low, avg_90d,
      volatility_score, seller_risk, price_series
    """
    params = {"key": api_key, "domain": 1, "asin": asin, "history": 1, "stats": 90}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{KEEPA_BASE}/product", params=params)
        resp.raise_for_status()
        data = resp.json()

    products = data.get("products")
    if not products:
        raise ValueError("Product not found on Keepa")

    product = products[0]

    # Title & image
    title = product.get("title", "Unknown Product")
    image_url = ""
    images = product.get("imagesCSV")
    if images:
        first = images.split(",")[0]
        image_url = f"https://images-na.ssl-images-amazon.com/images/I/{first}"

    # Price CSV — index 0 = Amazon price, index 1 = marketplace/new
    csv_amazon = product.get("csv", [None])[0] or []
    csv_new = product.get("csv", [None, None])[1] or []

    series_amazon = _parse_keepa_csv(csv_amazon)
    series_new = _parse_keepa_csv(csv_new)

    # Merge — prefer Amazon prices, fall back to marketplace
    series = series_amazon if series_amazon else series_new
    if not series:
        series = []

    # Stats block
    stats = product.get("stats", {})
    current_raw = stats.get("current", [None])[0]
    if current_raw is None or current_raw < 0:
        # try marketplace
        current_list = stats.get("current", [])
        current_raw = current_list[1] if len(current_list) > 1 and current_list[1] and current_list[1] > 0 else None

    current_price = round(current_raw / 100, 2) if current_raw and current_raw > 0 else None

    # If still no current price, use last point in series
    if current_price is None and series:
        current_price = series[-1]["price"]

    # Historical low & avg from stats
    min_prices = stats.get("min", [])
    avg_prices = stats.get("avg90", [])

    hist_low_raw = min_prices[0] if min_prices and min_prices[0] and min_prices[0] > 0 else None
    if hist_low_raw is None and len(min_prices) > 1:
        hist_low_raw = min_prices[1] if min_prices[1] and min_prices[1] > 0 else None
    hist_low = round(hist_low_raw / 100, 2) if hist_low_raw else None

    avg_90d_raw = avg_prices[0] if avg_prices and avg_prices[0] and avg_prices[0] > 0 else None
    if avg_90d_raw is None and len(avg_prices) > 1:
        avg_90d_raw = avg_prices[1] if avg_prices[1] and avg_prices[1] > 0 else None
    avg_90d = round(avg_90d_raw / 100, 2) if avg_90d_raw else None

    # Compute hist_low / avg from series if stats didn't provide them
    if series:
        prices = [p["price"] for p in series]
        if hist_low is None:
            hist_low = min(prices)
        if avg_90d is None:
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=90)
            recent = [p["price"] for p in series if p["date"] >= cutoff.strftime("%Y-%m-%d")]
            avg_90d = round(sum(recent) / len(recent), 2) if recent else round(sum(prices) / len(prices), 2)

    # Volatility — normalised std-dev of last 90 days
    volatility_score = 0.0
    if series:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=90)
        recent = [p["price"] for p in series if p["date"] >= cutoff.strftime("%Y-%m-%d")]
        if len(recent) >= 2 and avg_90d and avg_90d > 0:
            variance = sum((x - avg_90d) ** 2 for x in recent) / len(recent)
            std = math.sqrt(variance)
            volatility_score = round(std / avg_90d, 4)

    # Seller risk heuristic — many marketplace sellers or no Amazon direct = higher risk
    seller_risk = "low"
    offers = product.get("offers", [])
    amazon_direct = any(o.get("isFBA") or o.get("isPrime") for o in offers) if offers else False
    if not amazon_direct and not csv_amazon:
        seller_risk = "high"
    elif not amazon_direct:
        seller_risk = "medium"

    return {
        "title": title,
        "image": image_url,
        "current_price": current_price,
        "currency": "USD",
        "hist_low": hist_low,
        "avg_90d": avg_90d,
        "volatility_score": volatility_score,
        "seller_risk": seller_risk,
        "price_series": series,
    }


# ---------------------------------------------------------------------------
# Keepa search (by keyword)
# ---------------------------------------------------------------------------

async def search_keepa(query: str, api_key: str) -> list[dict]:
    """Search Keepa for products matching a keyword.  Returns list of {asin, title, image}."""
    params = {"key": api_key, "domain": 1, "type": "product", "term": query}
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"{KEEPA_BASE}/search", params=params)
        resp.raise_for_status()
        data = resp.json()
    results = []
    for asin in (data.get("asinList") or [])[:10]:
        results.append({"asin": asin, "title": None, "image": None})
    return results


# ---------------------------------------------------------------------------
# Amazon PA-API (optional)
# ---------------------------------------------------------------------------

def _pa_api_available() -> bool:
    return all(os.getenv(k) for k in ("AMAZON_ACCESS_KEY", "AMAZON_SECRET_KEY", "AMAZON_PARTNER_TAG"))


def _sign_pa_request(payload_bytes: bytes, target: str) -> dict[str, str]:
    """AWS Signature V4 for PA-API."""
    access = os.environ["AMAZON_ACCESS_KEY"]
    secret = os.environ["AMAZON_SECRET_KEY"]
    region = os.getenv("AMAZON_REGION", "us-east-1")
    host = f"webservices.amazon.com"
    service = "ProductAdvertisingAPI"
    now = datetime.now(tz=timezone.utc)
    datestamp = now.strftime("%Y%m%d")
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    content_type = "application/json; charset=UTF-8"

    canonical_uri = "/paapi5/getitems" if "GetItems" in target else "/paapi5/searchitems"

    canonical_headers = (
        f"content-encoding:amz-1.0\n"
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-amz-date:{amz_date}\n"
        f"x-amz-target:com.amazon.paapi5.v1.ProductAdvertisingAPIv1.{target}\n"
    )
    signed_headers = "content-encoding;content-type;host;x-amz-date;x-amz-target"
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    canonical_request = f"POST\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode()).hexdigest()}"

    def _sign(key, msg):
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = _sign(("AWS4" + secret).encode("utf-8"), datestamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = f"AWS4-HMAC-SHA256 Credential={access}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"

    return {
        "content-encoding": "amz-1.0",
        "content-type": content_type,
        "host": host,
        "x-amz-date": amz_date,
        "x-amz-target": f"com.amazon.paapi5.v1.ProductAdvertisingAPIv1.{target}",
        "Authorization": authorization,
    }


async def enrich_from_amazon(asin: str) -> dict[str, Any] | None:
    """Fetch title, image, and similar products from Amazon PA-API."""
    if not _pa_api_available():
        return None
    partner_tag = os.environ["AMAZON_PARTNER_TAG"]
    region = os.getenv("AMAZON_REGION", "us-east-1")
    host = "webservices.amazon.com"
    payload = {
        "ItemIds": [asin],
        "Resources": [
            "ItemInfo.Title",
            "Images.Primary.Large",
            "Offers.Listings.Price",
            "SimilarProducts",
        ],
        "PartnerTag": partner_tag,
        "PartnerType": "Associates",
        "Marketplace": "www.amazon.com",
    }
    payload_bytes = json.dumps(payload).encode()
    headers = _sign_pa_request(payload_bytes, "GetItems")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"https://{host}/paapi5/getitems",
                content=payload_bytes,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        item = data.get("ItemsResult", {}).get("Items", [{}])[0]
        title = item.get("ItemInfo", {}).get("Title", {}).get("DisplayValue")
        image = item.get("Images", {}).get("Primary", {}).get("Large", {}).get("URL")
        similar = []
        for s in item.get("SimilarProducts", [])[:6]:
            similar.append({"asin": s.get("ASIN"), "title": s.get("Title")})
        return {"title": title, "image": image, "similar": similar}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fiduciary decision engine
# ---------------------------------------------------------------------------

def compute_decision(
    current_price: float | None,
    hist_low: float | None,
    avg_90d: float | None,
    volatility_score: float,
    seller_risk: str,
) -> dict[str, Any]:
    """
    Deterministic, explainable decision engine.
    Returns {decision, confidence, explanation[]}.
    """
    if current_price is None or hist_low is None or avg_90d is None:
        return {
            "decision": "CLIP",
            "confidence": 30,
            "explanation": ["Insufficient pricing data to make a confident recommendation. Monitor this product."],
        }

    explanation: list[str] = []
    decision = "CLIP"
    confidence = 50

    ratio_to_low = current_price / hist_low if hist_low > 0 else 999
    ratio_to_avg = current_price / avg_90d if avg_90d > 0 else 999
    low_vol = volatility_score < 0.08

    # Strong TICK — at or near all-time low
    if ratio_to_low <= 1.05:
        decision = "TICK"
        confidence = 92
        explanation.append(
            f"Current price ${current_price:.2f} is within 5% of the all-time low ${hist_low:.2f} — excellent deal."
        )

    # TICK — below 90-day average with low volatility
    elif ratio_to_avg <= 0.95 and low_vol:
        decision = "TICK"
        confidence = 80
        explanation.append(
            f"Current price ${current_price:.2f} is {((1 - ratio_to_avg) * 100):.0f}% below the 90-day average ${avg_90d:.2f} with low volatility ({volatility_score:.2%})."
        )

    # SKIP — significantly above average or well above historic low
    elif ratio_to_avg >= 1.10 or ratio_to_low >= 1.25:
        decision = "SKIP"
        confidence = 78
        if ratio_to_avg >= 1.10:
            explanation.append(
                f"Current price ${current_price:.2f} is {((ratio_to_avg - 1) * 100):.0f}% above the 90-day average ${avg_90d:.2f} — overpriced."
            )
        if ratio_to_low >= 1.25:
            explanation.append(
                f"Current price ${current_price:.2f} is {((ratio_to_low - 1) * 100):.0f}% above the historic low ${hist_low:.2f} — wait for a drop."
            )

    # CLIP — near the average
    else:
        decision = "CLIP"
        confidence = 60
        explanation.append(
            f"Current price ${current_price:.2f} is near the 90-day average ${avg_90d:.2f} (within ±10%). Consider monitoring."
        )

    # Volatility commentary
    if volatility_score >= 0.15:
        explanation.append(
            f"High price volatility ({volatility_score:.2%}) — prices swing often. Waiting could pay off."
        )
        if decision == "TICK":
            decision = "CLIP"
            confidence = max(confidence - 15, 40)
            explanation.append("Downgraded from TICK to CLIP due to high volatility.")
    elif low_vol:
        explanation.append(f"Low price volatility ({volatility_score:.2%}) — price is stable.")

    # Seller risk
    if seller_risk == "high":
        explanation.append("Seller risk is HIGH — no Amazon-direct fulfillment detected. Exercise caution.")
        if decision == "TICK":
            decision = "CLIP"
            confidence = max(confidence - 10, 35)
            explanation.append("Downgraded from TICK to CLIP due to seller risk.")
        elif decision == "CLIP":
            decision = "SKIP"
            confidence = max(confidence - 5, 30)
            explanation.append("Downgraded from CLIP to SKIP due to seller risk.")
    elif seller_risk == "medium":
        explanation.append("Seller risk is MEDIUM — Amazon may not be the direct seller.")

    return {"decision": decision, "confidence": confidence, "explanation": explanation}


# ---------------------------------------------------------------------------
# DuckDuckGo web search (HTML scraping — no API key needed)
# ---------------------------------------------------------------------------

async def _ddg_search(query: str, max_results: int = 6) -> list[dict]:
    """Scrape DuckDuckGo HTML search results."""
    results: list[dict] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=headers,
            )
            resp.raise_for_status()
            html = resp.text

        # Simple parsing — extract result blocks
        import re
        # Each result link
        links = re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', html)
        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)

        for idx, (url, raw_title) in enumerate(links[:max_results]):
            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            snippet = ""
            if idx < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[idx]).strip()
            # Extract source domain
            from urllib.parse import urlparse
            source = urlparse(url).netloc.replace("www.", "")
            results.append({"title": title, "url": url, "snippet": snippet, "source": source})
    except Exception:
        pass
    return results


async def fetch_alternatives(product_title: str) -> list[dict]:
    """Search for alternative products."""
    query = f"{product_title} alternatives best buy"
    return await _ddg_search(query, 6)


async def fetch_diy_articles(product_title: str) -> list[dict]:
    """Search for DIY / repair / substitute articles."""
    query = f"{product_title} DIY repair substitute guide"
    return await _ddg_search(query, 6)
