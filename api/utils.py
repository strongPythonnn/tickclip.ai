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

    # Price manipulation analysis
    manipulation = detect_price_manipulation(series, current_price, hist_low, avg_90d)

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
        "price_manipulation": manipulation,
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
# Price manipulation detection
# ---------------------------------------------------------------------------

def detect_price_manipulation(
    series: list[dict],
    current_price: float | None,
    hist_low: float | None,
    avg_90d: float | None,
) -> dict[str, Any]:
    """
    Analyze price history for seller manipulation tactics:
    1. Inflate-then-discount: price spiked recently then "dropped" to seem like a deal
    2. Artificial reference pricing: current "sale" is actually the normal price
    3. Yo-yo pricing: repeated spikes/drops to create urgency
    4. Pre-event inflation: price raised before major sale events (Prime Day, Black Friday)
    5. Creeping inflation: slow gradual increases over months

    Returns {
      risk_level: "none" | "low" | "medium" | "high",
      score: 0-100 (higher = more manipulation detected),
      tactics: [{tactic, description, evidence, severity}],
      is_fake_deal: bool,
      true_market_price: float | None,
      inflated_by_pct: float | None,
    }
    """
    result: dict[str, Any] = {
        "risk_level": "none",
        "score": 0,
        "tactics": [],
        "is_fake_deal": False,
        "true_market_price": None,
        "inflated_by_pct": None,
    }

    if not series or len(series) < 5 or current_price is None:
        return result

    prices = [p["price"] for p in series]
    dates = [p["date"] for p in series]
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    manipulation_score = 0
    tactics: list[dict] = []

    # ── Helpers ──
    cutoff_30 = (datetime.now(tz=timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    cutoff_60 = (datetime.now(tz=timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")
    cutoff_90 = (datetime.now(tz=timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    cutoff_180 = (datetime.now(tz=timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")

    recent_30 = [p for p, d in zip(prices, dates) if d >= cutoff_30]
    recent_60 = [p for p, d in zip(prices, dates) if d >= cutoff_60]
    recent_90 = [p for p, d in zip(prices, dates) if d >= cutoff_90]
    older_half = [p for p, d in zip(prices, dates) if d < cutoff_90]

    # Long-term median (robust against spikes)
    sorted_all = sorted(prices)
    median_all = sorted_all[len(sorted_all) // 2]

    # ── TACTIC 1: Inflate-then-discount ──
    # Price spiked significantly in the last 60 days then dropped to current
    if recent_60 and len(recent_60) >= 3:
        max_60 = max(recent_60)
        if max_60 > median_all * 1.30 and current_price < max_60 * 0.85:
            pct_spike = round((max_60 / median_all - 1) * 100)
            pct_drop = round((1 - current_price / max_60) * 100)
            severity = "high" if pct_spike >= 50 else "medium"
            manipulation_score += 35 if severity == "high" else 20
            tactics.append({
                "tactic": "Inflate-then-Discount",
                "description": f"Price was inflated to ${max_60:.2f} ({pct_spike}% above median ${median_all:.2f}), then \"dropped\" {pct_drop}% to the current ${current_price:.2f}. The sale looks big but the spike was artificial.",
                "evidence": f"60-day peak ${max_60:.2f} vs. historic median ${median_all:.2f}",
                "severity": severity,
            })

    # ── TACTIC 2: Fake reference price / "sale" is the real price ──
    # Current price is actually the most common price (mode)
    if len(prices) >= 10:
        # Bucket prices to nearest dollar to find mode
        buckets: dict[int, int] = {}
        for p in prices:
            b = round(p)
            buckets[b] = buckets.get(b, 0) + 1
        mode_price = max(buckets, key=buckets.get)
        mode_freq = buckets[mode_price]
        mode_pct = mode_freq / len(prices)

        if abs(current_price - mode_price) <= 2 and mode_pct >= 0.25:
            # Current price IS the most common price — any "sale" framing is fake
            if avg_90d and current_price >= avg_90d * 0.97:
                manipulation_score += 15
                tactics.append({
                    "tactic": "Fake Sale",
                    "description": f"The current price ${current_price:.2f} is essentially the regular price (most common: ${mode_price}, seen {mode_pct:.0%} of the time). Any \"sale\" or \"discount\" label is misleading.",
                    "evidence": f"${mode_price} appears in {mode_pct:.0%} of price history",
                    "severity": "medium",
                })

    # ── TACTIC 3: Yo-yo pricing (repeated spikes to create urgency) ──
    if len(recent_90) >= 6:
        direction_changes = 0
        for i in range(2, len(recent_90)):
            prev_dir = recent_90[i-1] - recent_90[i-2]
            curr_dir = recent_90[i] - recent_90[i-1]
            if (prev_dir > 0 and curr_dir < 0) or (prev_dir < 0 and curr_dir > 0):
                # Only count significant changes (>5% of price)
                if abs(curr_dir) > current_price * 0.05:
                    direction_changes += 1
        if direction_changes >= 4:
            manipulation_score += 25
            tactics.append({
                "tactic": "Yo-Yo Pricing",
                "description": f"Price changed direction {direction_changes} times in 90 days with swings >5%. This creates false urgency — \"buy now before it goes up again.\"",
                "evidence": f"{direction_changes} significant reversals in 90 days",
                "severity": "high",
            })
        elif direction_changes >= 2:
            manipulation_score += 10
            tactics.append({
                "tactic": "Yo-Yo Pricing",
                "description": f"Price changed direction {direction_changes} times in 90 days. Moderate price instability that may be used to create urgency.",
                "evidence": f"{direction_changes} reversals in 90 days",
                "severity": "low",
            })

    # ── TACTIC 4: Pre-event inflation ──
    # Check if there was a spike before known sale events
    sale_events = [
        ("07-01", "07-20", "Prime Day"),
        ("11-15", "11-30", "Black Friday"),
        ("11-25", "12-02", "Cyber Monday"),
    ]
    for start_m, end_m, event_name in sale_events:
        for yr_offset in range(2):
            year = datetime.now(tz=timezone.utc).year - yr_offset
            pre_start = f"{year}-{start_m}"
            event_end = f"{year}-{end_m}"
            # 30 days before the event
            pre_month = (datetime(year, int(start_m.split("-")[0]), 1, tzinfo=timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")

            pre_prices = [p for p, d in zip(prices, dates) if pre_month <= d < pre_start]
            event_prices = [p for p, d in zip(prices, dates) if pre_start <= d <= event_end]

            if pre_prices and event_prices:
                avg_pre = sum(pre_prices) / len(pre_prices)
                max_pre = max(pre_prices)
                min_event = min(event_prices)
                # Was price raised before the event, then "discounted" during it?
                if max_pre > avg_pre * 1.15 and min_event < max_pre * 0.80:
                    inflation_pct = round((max_pre / avg_pre - 1) * 100)
                    manipulation_score += 20
                    tactics.append({
                        "tactic": "Pre-Event Inflation",
                        "description": f"Price was raised {inflation_pct}% before {event_name} {year} (to ${max_pre:.2f}), then \"discounted\" during the event to ${min_event:.2f}. The deal was manufactured.",
                        "evidence": f"Pre-event avg ${avg_pre:.2f} → spike ${max_pre:.2f} → event \"sale\" ${min_event:.2f}",
                        "severity": "high",
                    })
                    break  # one example per event is enough

    # ── TACTIC 5: Creeping inflation ──
    if older_half and recent_30:
        avg_old = sum(older_half) / len(older_half)
        avg_recent = sum(recent_30) / len(recent_30)
        if avg_recent > avg_old * 1.15 and avg_old > 0:
            creep_pct = round((avg_recent / avg_old - 1) * 100)
            manipulation_score += 15
            tactics.append({
                "tactic": "Creeping Inflation",
                "description": f"Average price has gradually risen {creep_pct}% (from ${avg_old:.2f} to ${avg_recent:.2f}). Small increases over time are harder to notice but add up.",
                "evidence": f"Older avg ${avg_old:.2f} → Recent avg ${avg_recent:.2f} (+{creep_pct}%)",
                "severity": "medium" if creep_pct < 25 else "high",
            })

    # ── Compute true market price ──
    # Use the median excluding top-10% spike prices
    cutoff_idx = max(1, int(len(sorted_all) * 0.90))
    trimmed = sorted_all[:cutoff_idx]
    true_market = round(sum(trimmed) / len(trimmed), 2) if trimmed else median_all

    inflated_by = None
    if current_price > true_market and true_market > 0:
        inflated_by = round((current_price / true_market - 1) * 100, 1)

    # ── Final scoring ──
    manipulation_score = min(manipulation_score, 100)
    if manipulation_score >= 50:
        risk_level = "high"
    elif manipulation_score >= 25:
        risk_level = "medium"
    elif manipulation_score > 0:
        risk_level = "low"
    else:
        risk_level = "none"

    is_fake_deal = any(t["tactic"] in ("Inflate-then-Discount", "Fake Sale", "Pre-Event Inflation")
                       for t in tactics if t.get("severity") in ("high", "medium"))

    result["risk_level"] = risk_level
    result["score"] = manipulation_score
    result["tactics"] = tactics
    result["is_fake_deal"] = is_fake_deal
    result["true_market_price"] = true_market
    result["inflated_by_pct"] = inflated_by

    return result


# ---------------------------------------------------------------------------
# Fiduciary decision engine
# ---------------------------------------------------------------------------

def compute_decision(
    current_price: float | None,
    hist_low: float | None,
    avg_90d: float | None,
    volatility_score: float,
    seller_risk: str,
    manipulation: dict | None = None,
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

    # Price manipulation
    if manipulation and manipulation.get("risk_level") in ("medium", "high"):
        manip_score = manipulation.get("score", 0)
        is_fake = manipulation.get("is_fake_deal", False)
        true_price = manipulation.get("true_market_price")

        if is_fake:
            explanation.append(
                f"PRICE MANIPULATION DETECTED: This deal appears artificially inflated. "
                f"True market price is ~${true_price:.2f}." if true_price else
                "PRICE MANIPULATION DETECTED: This deal appears artificially inflated."
            )
            if decision == "TICK":
                decision = "CLIP"
                confidence = max(confidence - 20, 30)
                explanation.append("Downgraded from TICK to CLIP due to suspected price manipulation.")
            elif decision == "CLIP":
                confidence = max(confidence - 10, 25)
        elif manipulation["risk_level"] == "high":
            explanation.append(
                "High price manipulation risk detected — seller pricing patterns are suspicious."
            )
            if decision == "TICK":
                decision = "CLIP"
                confidence = max(confidence - 15, 35)
                explanation.append("Downgraded from TICK to CLIP due to manipulation risk.")
        else:
            explanation.append("Moderate price manipulation signals detected — review the pricing history carefully.")

        if manipulation.get("inflated_by_pct") and manipulation["inflated_by_pct"] > 5:
            explanation.append(
                f"Current price may be inflated ~{manipulation['inflated_by_pct']:.0f}% above true market value (${true_price:.2f})." if true_price else
                f"Current price may be inflated ~{manipulation['inflated_by_pct']:.0f}% above true market value."
            )

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
