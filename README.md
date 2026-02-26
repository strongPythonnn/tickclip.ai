# TickClip.ai

Fiduciary AI shopping decision platform. Evaluates Amazon products and tells you: **TICK** (buy), **CLIP** (monitor), or **SKIP** (avoid).

## Quick Start

```bash
pip install -r api/requirements.txt
export KEEPA_API_KEY=your_key
uvicorn api.server:app --reload
```

Open http://localhost:8000.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `KEEPA_API_KEY` | Yes | Keepa API key |
| `AMAZON_ACCESS_KEY` | No | Amazon PA-API access key |
| `AMAZON_SECRET_KEY` | No | Amazon PA-API secret key |
| `AMAZON_PARTNER_TAG` | No | Amazon Associates partner tag |
| `AMAZON_REGION` | No | Amazon region (default: us-east-1) |

## Deploy on Render

Connect this repo and Render will auto-detect `render.yaml`. Set `KEEPA_API_KEY` in the Render dashboard.

## How It Works

1. Search by product name or paste an Amazon ASIN/URL
2. Keepa provides price history data
3. Deterministic decision engine evaluates the deal
4. For CLIP/SKIP decisions, alternatives and DIY articles are fetched

No affiliate links. No commission. Fiduciary-first.
