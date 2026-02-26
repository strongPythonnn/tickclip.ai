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

    def _safe_int(val) -> int | None:
        """Extract a positive integer from a value that may be int, list, or None."""
        if isinstance(val, list):
            # Keepa sometimes nests lists — dig into first scalar
            for item in val:
                result = _safe_int(item)
                if result is not None:
                    return result
            return None
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
        return None

    def _pick_stat(stat_list, *indices) -> int | None:
        """Try each index in stat_list, return first positive int."""
        if not isinstance(stat_list, list):
            return _safe_int(stat_list)
        for idx in indices:
            if idx < len(stat_list):
                result = _safe_int(stat_list[idx])
                if result is not None:
                    return result
        return None

    current_raw = _pick_stat(stats.get("current", []), 0, 1)
    current_price = round(current_raw / 100, 2) if current_raw else None

    # If still no current price, use last point in series
    if current_price is None and series:
        current_price = series[-1]["price"]

    # Historical low & avg from stats
    hist_low_raw = _pick_stat(stats.get("min", []), 0, 1)
    hist_low = round(hist_low_raw / 100, 2) if hist_low_raw else None

    avg_90d_raw = _pick_stat(stats.get("avg90", []), 0, 1)
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

import re as _re
from urllib.parse import urlparse as _urlparse
import asyncio


_DDG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


async def _ddg_search(query: str, max_results: int = 8) -> list[dict]:
    """Scrape DuckDuckGo HTML search results."""
    results: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=_DDG_HEADERS,
            )
            resp.raise_for_status()
            html = resp.text

        links = _re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', html)
        snippets = _re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, _re.DOTALL)

        for idx, (url, raw_title) in enumerate(links[:max_results]):
            title = _re.sub(r"<[^>]+>", "", raw_title).strip()
            snippet = ""
            if idx < len(snippets):
                snippet = _re.sub(r"<[^>]+>", "", snippets[idx]).strip()
            source = _urlparse(url).netloc.replace("www.", "")
            results.append({"title": title, "url": url, "snippet": snippet, "source": source})
    except Exception:
        pass
    return results


# ---------------------------------------------------------------------------
# Multi-source retailer & deal searches
# ---------------------------------------------------------------------------

# Retailer-specific site searches
_RETAILER_QUERIES = {
    "walmart": "site:walmart.com {product} buy",
    "bestbuy": "site:bestbuy.com {product}",
    "target": "site:target.com {product}",
    "costco": "site:costco.com {product}",
    "newegg": "site:newegg.com {product}",
}

# Deal site searches
_DEAL_QUERIES = {
    "slickdeals": "site:slickdeals.net {product} deal",
    "retailmenot": "site:retailmenot.com {product} coupon",
    "groupon": "site:groupon.com {product}",
    "camelcamelcamel": "site:camelcamelcamel.com {product}",
    "honey": "site:joinhoney.com {product}",
}

# Retailer display names & colors (sent to frontend)
RETAILER_META = {
    "walmart.com": {"name": "Walmart", "color": "#0071dc"},
    "bestbuy.com": {"name": "Best Buy", "color": "#0046be"},
    "target.com": {"name": "Target", "color": "#cc0000"},
    "costco.com": {"name": "Costco", "color": "#e31837"},
    "newegg.com": {"name": "Newegg", "color": "#f7821b"},
    "amazon.com": {"name": "Amazon", "color": "#ff9900"},
    "slickdeals.net": {"name": "Slickdeals", "color": "#2e8540"},
    "retailmenot.com": {"name": "RetailMeNot", "color": "#e22a2a"},
    "groupon.com": {"name": "Groupon", "color": "#53a318"},
    "camelcamelcamel.com": {"name": "CamelCamelCamel", "color": "#884499"},
    "joinhoney.com": {"name": "Honey", "color": "#ff6801"},
    "ebay.com": {"name": "eBay", "color": "#e53238"},
}


def _enrich_result(item: dict, category: str) -> dict:
    """Add retailer metadata and category tag to a search result."""
    domain = item.get("source", "")
    # Match against known retailers
    meta = None
    for key, val in RETAILER_META.items():
        if key in domain:
            meta = val
            break
    item["retailer_name"] = meta["name"] if meta else domain.split(".")[0].title()
    item["retailer_color"] = meta["color"] if meta else "#6b7280"
    item["category"] = category  # "retailer", "deal", "alternative", "diy"
    return item


async def _search_source(query_template: str, product: str, category: str, max_results: int = 3) -> list[dict]:
    """Run a single site-scoped search and tag results."""
    query = query_template.format(product=product)
    results = await _ddg_search(query, max_results)
    return [_enrich_result(r, category) for r in results]


async def fetch_retailer_prices(product_title: str) -> list[dict]:
    """
    Search across Walmart, Best Buy, Target, Costco, Newegg for the product.
    Returns tagged results from each retailer.
    """
    tasks = [
        _search_source(tpl, product_title, "retailer", 2)
        for tpl in _RETAILER_QUERIES.values()
    ]
    groups = await asyncio.gather(*tasks, return_exceptions=True)
    combined: list[dict] = []
    for group in groups:
        if isinstance(group, list):
            combined.extend(group)
    return combined


async def fetch_deals(product_title: str) -> list[dict]:
    """
    Search Slickdeals, RetailMeNot, Groupon, CamelCamelCamel, Honey.
    Returns tagged deal results.
    """
    tasks = [
        _search_source(tpl, product_title, "deal", 2)
        for tpl in _DEAL_QUERIES.values()
    ]
    groups = await asyncio.gather(*tasks, return_exceptions=True)
    combined: list[dict] = []
    for group in groups:
        if isinstance(group, list):
            combined.extend(group)
    return combined


async def fetch_alternatives(product_title: str) -> list[dict]:
    """Search for alternative products across the web."""
    queries = [
        f"{product_title} best alternative 2025",
        f"{product_title} vs competitor comparison",
    ]
    tasks = [_ddg_search(q, 4) for q in queries]
    groups = await asyncio.gather(*tasks, return_exceptions=True)
    combined: list[dict] = []
    seen_urls: set[str] = set()
    for group in groups:
        if isinstance(group, list):
            for item in group:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    combined.append(_enrich_result(item, "alternative"))
    return combined[:8]


async def fetch_diy_articles(product_title: str) -> list[dict]:
    """Search for DIY / repair / substitute articles."""
    queries = [
        f"{product_title} DIY repair guide",
        f"{product_title} fix yourself instead of buying",
        f"{product_title} substitute homemade",
    ]
    tasks = [_ddg_search(q, 3) for q in queries]
    groups = await asyncio.gather(*tasks, return_exceptions=True)
    combined: list[dict] = []
    seen_urls: set[str] = set()
    for group in groups:
        if isinstance(group, list):
            for item in group:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    combined.append(_enrich_result(item, "diy"))
    return combined[:8]


# ---------------------------------------------------------------------------
# Review analysis — scrape expert & community reviews, compute sentiment
# ---------------------------------------------------------------------------

# Trusted review sources with site-scoped searches
_REVIEW_QUERIES = [
    ("site:reddit.com {product} review worth it", "reddit"),
    ("site:youtube.com {product} review", "youtube"),
    ("{product} review wirecutter OR rtings OR tomsguide OR consumerreports", "expert"),
    ("{product} review problems complaints", "critical"),
]

# Sentiment keyword dictionaries
_POSITIVE_WORDS = {
    "excellent", "amazing", "perfect", "love", "great", "best", "fantastic",
    "awesome", "solid", "reliable", "recommend", "worth", "impressed",
    "quality", "premium", "durable", "comfortable", "fast", "smooth",
    "beautiful", "outstanding", "superb", "flawless", "incredible",
    "favorite", "happy", "pleased", "satisfied", "good", "nice",
    "well-built", "sturdy", "upgrade", "improved", "bargain", "value",
}
_NEGATIVE_WORDS = {
    "terrible", "awful", "worst", "hate", "bad", "cheap", "broken",
    "defective", "flimsy", "disappointing", "waste", "overpriced",
    "avoid", "return", "refund", "poor", "fails", "died", "stopped",
    "unreliable", "junk", "garbage", "useless", "regret", "problem",
    "issue", "complaint", "fragile", "slow", "loud", "uncomfortable",
    "malfunction", "scam", "ripoff", "horrible", "sucks", "worse",
}

# Review source metadata
_REVIEW_SOURCE_META = {
    "reddit.com": {"name": "Reddit", "color": "#ff4500", "icon": "community"},
    "youtube.com": {"name": "YouTube", "color": "#ff0000", "icon": "video"},
    "wirecutter.com": {"name": "Wirecutter", "color": "#0a6abf", "icon": "expert"},
    "rtings.com": {"name": "RTINGS", "color": "#1a73e8", "icon": "expert"},
    "tomsguide.com": {"name": "Tom's Guide", "color": "#e4002b", "icon": "expert"},
    "consumerreports.org": {"name": "Consumer Reports", "color": "#007749", "icon": "expert"},
    "techradar.com": {"name": "TechRadar", "color": "#0080ff", "icon": "expert"},
    "cnet.com": {"name": "CNET", "color": "#d41e1e", "icon": "expert"},
    "verge.com": {"name": "The Verge", "color": "#e5127d", "icon": "expert"},
    "pcmag.com": {"name": "PCMag", "color": "#ed1c24", "icon": "expert"},
}


def _analyze_sentiment(text: str) -> dict:
    """Keyword-based sentiment scoring for a snippet of text."""
    words = set(_re.findall(r"[a-z]+", text.lower()))
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return {"score": 0.5, "label": "neutral", "positive": 0, "negative": 0}
    score = round(pos / total, 2)
    if score >= 0.65:
        label = "positive"
    elif score <= 0.35:
        label = "negative"
    else:
        label = "mixed"
    return {"score": score, "label": label, "positive": pos, "negative": neg}


def _tag_review(item: dict, query_type: str) -> dict:
    """Add review-specific metadata to a search result."""
    domain = item.get("source", "")
    meta = None
    for key, val in _REVIEW_SOURCE_META.items():
        if key in domain:
            meta = val
            break
    item["review_source"] = meta["name"] if meta else domain.split(".")[0].title()
    item["review_color"] = meta["color"] if meta else "#6b7280"
    item["review_type"] = meta["icon"] if meta else "general"
    item["query_type"] = query_type  # "reddit", "youtube", "expert", "critical"

    # Sentiment on the snippet
    snippet = item.get("snippet", "") + " " + item.get("title", "")
    item["sentiment"] = _analyze_sentiment(snippet)
    return item


async def fetch_reviews(product_title: str) -> dict:
    """
    Fetch reviews from Reddit, YouTube, expert sites, and critical reviews.
    Returns:
      {
        reviews: [{title, url, snippet, source, review_source, review_color,
                   review_type, query_type, sentiment}],
        summary: {
          total, positive_count, negative_count, mixed_count,
          overall_score, overall_label,
          top_pros[], top_cons[], verdict
        }
      }
    """
    tasks = [
        _ddg_search(tpl.format(product=product_title), 4)
        for tpl, _ in _REVIEW_QUERIES
    ]
    groups = await asyncio.gather(*tasks, return_exceptions=True)

    all_reviews: list[dict] = []
    seen: set[str] = set()
    for idx, group in enumerate(groups):
        if not isinstance(group, list):
            continue
        query_type = _REVIEW_QUERIES[idx][1]
        for item in group:
            if item["url"] not in seen:
                seen.add(item["url"])
                all_reviews.append(_tag_review(item, query_type))

    # Aggregate sentiment
    sentiments = [r["sentiment"] for r in all_reviews if r.get("sentiment")]
    pos_count = sum(1 for s in sentiments if s["label"] == "positive")
    neg_count = sum(1 for s in sentiments if s["label"] == "negative")
    mix_count = sum(1 for s in sentiments if s["label"] == "mixed")
    neutral_count = sum(1 for s in sentiments if s["label"] == "neutral")
    total = len(sentiments)

    if total > 0:
        avg_score = round(sum(s["score"] for s in sentiments) / total, 2)
    else:
        avg_score = 0.5

    if avg_score >= 0.65:
        overall_label = "Mostly Positive"
    elif avg_score >= 0.45:
        overall_label = "Mixed"
    else:
        overall_label = "Mostly Negative"

    # Extract pros and cons from snippets
    all_text = " ".join(r.get("snippet", "") + " " + r.get("title", "") for r in all_reviews).lower()
    all_words = set(_re.findall(r"[a-z]+", all_text))
    found_pros = sorted(all_words & _POSITIVE_WORDS)[:5]
    found_cons = sorted(all_words & _NEGATIVE_WORDS)[:5]

    # Verdict
    if avg_score >= 0.7 and neg_count <= 1:
        verdict = "Reviewers overwhelmingly recommend this product."
    elif avg_score >= 0.55:
        verdict = "Reviews are generally positive with some concerns."
    elif avg_score >= 0.4:
        verdict = "Reviews are mixed — do more research before buying."
    else:
        verdict = "Reviews lean negative — consider alternatives."

    return {
        "reviews": all_reviews[:12],
        "summary": {
            "total": total,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "mixed_count": mix_count,
            "neutral_count": neutral_count,
            "overall_score": avg_score,
            "overall_label": overall_label,
            "top_pros": found_pros,
            "top_cons": found_cons,
            "verdict": verdict,
        },
    }
