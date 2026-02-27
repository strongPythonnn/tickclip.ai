"""
TickClip.ai — Fiduciary AI Agent

An unbiased AI shopping advisor that calls real data tools before making
any claims. Never pushes purchases. Always transparent about data sources.
"""

import json
import os
import time
from pathlib import Path

import anthropic

from api.market_config import MarketConfig, get_market
from api.utils import (
    _web_search,
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

# ---------------------------------------------------------------------------
# System prompt — enforces fiduciary duty
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the intelligence core of TickClip, a fiduciary AI decision-support system.

Your purpose is to protect the user's financial interest at the point of online purchase.

You are NOT a sales assistant.
You do NOT optimize for conversion.
You do NOT promote purchases.

You act as a neutral arbitration engine that analyzes pricing data, seller risk, and manipulation signals to minimize financial regret.

Your optimization goal is:
- Financial prudence
- Risk reduction
- Transparency
- Consumer outcome maximization

If data is incomplete, state uncertainty clearly.
Never fabricate data.
Never guess numbers.

You MUST call tools to get real data before making any claims. All analysis must be based strictly on provided data.

## TASKS

### 1. PRICE INTELLIGENCE ANALYSIS
When historical price data is provided, you must:
- Identify historical low (All-Time Low)
- Compare current price to 30, 90, and 180-day medians (if computable)
- Calculate price percentile within available history
- Detect abnormal price spikes before discounts
- Identify volatility patterns
- Estimate likelihood of price drop (only if statistically defensible)

Output format:
PRICE ANALYSIS
Current Price:
Historical Low:
90-Day Median:
Price Percentile:
Inflation vs Median:
Volatility Level: Low / Medium / High
Drop Probability: (if computable, otherwise state insufficient data)
Conclusion: Fair / Slightly Elevated / Inflated / Near Historical Low

### 2. MANIPULATION DETECTION
If signals indicate potential behavioral manipulation, evaluate:
- Pre-sale price inflation
- Artificial urgency framing
- Suspicious discount anchoring
- Review velocity anomalies

Output format:
MANIPULATION RISK
Pre-discount inflation detected: Yes / No
Urgency manipulation risk: Low / Medium / High
Review anomaly risk: Low / Medium / High
Overall manipulation score: 0-100 (only if justifiable)

If insufficient signals exist, state: "No manipulation signals detected in provided data."

### 3. SELLER TRUST EVALUATION
If seller data is provided, assess:
- Rating consistency
- Distribution of reviews
- Seller longevity
- Risk indicators

Output format:
SELLER TRUST SCORE
Score: 0-100 (only if computable)
Risk Level: Low / Moderate / Elevated
Explanation:

If no seller data is provided, state: "Seller data not available."

### 4. FIDUCIARY VERDICT
Synthesize all signals and produce one final arbitration:
- TICK: Price near historical low and low risk
- CLIP: Not optimal timing; monitor
- SKIP: Inflated price or elevated risk

Output format:
FINAL VERDICT: TICK / CLIP / SKIP
FIDUCIARY JUSTIFICATION: A concise, data-grounded explanation. No persuasion language.

### 5. PROTECTIVE MODE
If verdict is CLIP or SKIP, provide protective guidance:
- Suggested waiting window (if defensible)
- Risk explanation
- Neutral alternatives if data is available
- Cooling-off suggestion if repeated impulsive behavior detected

Do not push affiliate links. Do not exaggerate savings. Do not speculate beyond evidence.

## STYLE REQUIREMENTS
Tone: Analytical, calm, transparent.
Avoid marketing language.
Avoid emotional persuasion.
Avoid emojis.
Be concise but structured.
If uncertainty exists, say so clearly.

## TOOL USAGE
- "Should I buy X?" -> Call get_price_data + evaluate_product + analyze_reviews. Lead with verdict.
- "Is this a good deal?" -> Call get_price_data + compare_prices. Compare to historic low and other retailers.
- "What are alternatives?" -> Call find_alternatives. Compare features and prices.
- "Any deals on X?" -> Call find_deals + find_similar_deals. List active deals with sources.
- General questions -> Call web_search for latest information.

Always call multiple tools when needed. The user trusts you as their fiduciary guardian.

MISSION: Minimize consumer regret. Maximize rational decision-making. Act as a fiduciary guardian — not a retailer."""

# ---------------------------------------------------------------------------
# Tool definitions for Claude API
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_products",
        "description": "Search for products by keyword or name. Returns a list of ASINs with titles and images. Use this when the user asks about a product by name and you need to find its ASIN.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Product name or search keywords",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_price_data",
        "description": "Get detailed price history and metrics for a product by ASIN. Returns current price, historic low, 90-day average, volatility score, seller risk, and full price series. Essential for any buy/wait/skip recommendation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "asin": {
                    "type": "string",
                    "description": "Amazon ASIN (10-character product ID)",
                }
            },
            "required": ["asin"],
        },
    },
    {
        "name": "get_amazon_details",
        "description": "Get rich Amazon product details: offers with prices/savings, promotions, features, brand, seller info. Use alongside get_price_data for complete product picture.",
        "input_schema": {
            "type": "object",
            "properties": {
                "asin": {
                    "type": "string",
                    "description": "Amazon ASIN (10-character product ID)",
                }
            },
            "required": ["asin"],
        },
    },
    {
        "name": "evaluate_product",
        "description": "Run the deterministic TICK/CLIP/SKIP decision engine on price data. Returns decision, confidence percentage, and explanation. You MUST call get_price_data first to get the required inputs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "current_price": {"type": "number", "description": "Current price in USD"},
                "hist_low": {"type": "number", "description": "Historical low price in USD"},
                "avg_90d": {"type": "number", "description": "90-day average price in USD"},
                "volatility_score": {"type": "number", "description": "Price volatility 0-1"},
                "seller_risk": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Seller risk level",
                },
            },
            "required": [
                "current_price",
                "hist_low",
                "avg_90d",
                "volatility_score",
                "seller_risk",
            ],
        },
    },
    {
        "name": "compare_prices",
        "description": "Search Walmart, Best Buy, Target, Costco, Newegg for the product. Returns prices from each retailer and a price_insight summary showing the cheapest option.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_title": {
                    "type": "string",
                    "description": "Full product name to search for",
                },
                "amazon_price": {
                    "type": "number",
                    "description": "Current Amazon price for comparison (optional)",
                },
            },
            "required": ["product_title"],
        },
    },
    {
        "name": "find_deals",
        "description": "Search deal sites (Slickdeals, RetailMeNot, Groupon, CamelCamelCamel, Honey) for active deals and coupons on the product.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_title": {
                    "type": "string",
                    "description": "Full product name to search for deals",
                }
            },
            "required": ["product_title"],
        },
    },
    {
        "name": "analyze_reviews",
        "description": "Fetch and analyze reviews from Reddit, YouTube, expert sites, and critical reviews. Returns individual reviews with sentiment analysis and an overall summary with pros, cons, and verdict.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_title": {
                    "type": "string",
                    "description": "Full product name to search reviews for",
                }
            },
            "required": ["product_title"],
        },
    },
    {
        "name": "find_alternatives",
        "description": "Search for alternative/competitor products. Returns comparison articles and competitor mentions from trusted sources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_title": {
                    "type": "string",
                    "description": "Product name to find alternatives for",
                }
            },
            "required": ["product_title"],
        },
    },
    {
        "name": "find_diy_options",
        "description": "Search for DIY repair guides, fix-it-yourself articles, and substitute/homemade alternatives. Helps users save money by repairing instead of buying.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_title": {
                    "type": "string",
                    "description": "Product name to find DIY options for",
                }
            },
            "required": ["product_title"],
        },
    },
    {
        "name": "find_similar_deals",
        "description": "Search Amazon for deals on similar products. Returns products with prices, savings percentages, and Prime eligibility.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Product name or category to find deals for",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": "General web search via Google (Serper.dev). Use for questions not covered by other tools — latest news, specific comparisons, warranty info, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 5)",
                },
            },
            "required": ["query"],
        },
    },
]

# ---------------------------------------------------------------------------
# History store — learns from past interactions
# ---------------------------------------------------------------------------

HISTORY_DIR = Path(__file__).resolve().parent.parent / "data"


def _log_interaction(entry: dict) -> None:
    """Append an interaction to the history log."""
    HISTORY_DIR.mkdir(exist_ok=True)
    log_file = HISTORY_DIR / "agent_history.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _load_recent_history(n: int = 20) -> list[dict]:
    """Load the last N interactions for context."""
    log_file = HISTORY_DIR / "agent_history.jsonl"
    if not log_file.exists():
        return []
    lines = log_file.read_text().strip().split("\n")
    recent = lines[-n:] if len(lines) > n else lines
    entries = []
    for line in recent:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


# ---------------------------------------------------------------------------
# Tool execution — maps tool names to real functions
# ---------------------------------------------------------------------------


async def _execute_tool(name: str, args: dict) -> str:
    """Execute a tool and return the result as a JSON string."""
    keepa_key = os.getenv("KEEPA_API_KEY", "")

    try:
        if name == "search_products":
            result = await search_keepa(args["query"], keepa_key)
        elif name == "get_price_data":
            result = await fetch_keepa(args["asin"], keepa_key)
        elif name == "get_amazon_details":
            result = await enrich_from_amazon(args["asin"])
        elif name == "evaluate_product":
            result = compute_decision(
                args["current_price"],
                args["hist_low"],
                args["avg_90d"],
                args["volatility_score"],
                args["seller_risk"],
            )
        elif name == "compare_prices":
            result = await fetch_retailer_prices(
                args["product_title"], args.get("amazon_price")
            )
        elif name == "find_deals":
            result = await fetch_deals(args["product_title"])
        elif name == "analyze_reviews":
            result = await fetch_reviews(args["product_title"])
        elif name == "find_alternatives":
            result = await fetch_alternatives(args["product_title"])
        elif name == "find_diy_options":
            result = await fetch_diy_articles(args["product_title"])
        elif name == "find_similar_deals":
            result = await search_amazon_deals(args["query"])
        elif name == "web_search":
            result = await _web_search(args["query"], args.get("max_results", 5))
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": str(e)}

    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class FiduciaryAgent:
    """Unbiased AI shopping advisor powered by Claude with tool use."""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"

    async def chat(
        self,
        message: str,
        asin: str | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """
        Process a user message with tool use.

        Returns:
            {
                "reply": str,           # Agent's response text
                "tools_used": [str],    # Which tools were called (transparency)
                "data": dict,           # Any structured data from tools
            }
        """
        # Build messages
        messages = []

        # Add conversation history
        if history:
            for h in history[-10:]:  # Keep last 10 turns
                messages.append({"role": h["role"], "content": h["content"]})

        # Build user message with context
        user_content = message
        if asin:
            user_content = f"[Context: The user is looking at product ASIN {asin} on the evaluation page.]\n\n{message}"

        messages.append({"role": "user", "content": user_content})

        # Agent loop — let Claude call tools until it has enough data
        tools_used: list[str] = []
        tool_data: dict = {}
        max_turns = 8  # Safety limit

        for _ in range(max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Process all tool calls in this response
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tools_used.append(tool_name)

                        # Execute the tool
                        result_str = await _execute_tool(tool_name, tool_input)

                        # Store structured data
                        try:
                            tool_data[tool_name] = json.loads(result_str)
                        except json.JSONDecodeError:
                            tool_data[tool_name] = result_str

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_str,
                            }
                        )

                messages.append({"role": "user", "content": tool_results})
            else:
                # Claude is done — extract the text reply
                reply = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        reply += block.text

                # Log interaction for learning
                _log_interaction(
                    {
                        "timestamp": time.time(),
                        "user_message": message,
                        "asin": asin,
                        "tools_used": tools_used,
                        "reply_length": len(reply),
                    }
                )

                return {
                    "reply": reply,
                    "tools_used": list(set(tools_used)),
                    "data": tool_data,
                }

        # Safety: if we hit max turns, return what we have
        return {
            "reply": "I gathered a lot of data but hit my analysis limit. Please ask a more specific question and I'll focus my research.",
            "tools_used": list(set(tools_used)),
            "data": tool_data,
        }


# ---------------------------------------------------------------------------
# Standalone AI analysis — Price Manipulation
# ---------------------------------------------------------------------------

MANIPULATION_PROMPT = """You are TickClip's fiduciary price manipulation analyst. You act as a neutral arbitration engine — not a sales assistant.

Given the raw manipulation detection data for a product, write a concise analysis (3-5 sentences) in plain paragraph text.

Rules:
- Be analytical, calm, and transparent. Cite exact numbers (prices, percentages).
- If pre-discount inflation is detected, state it directly with evidence.
- If artificial urgency or suspicious discount anchoring exists, flag it.
- If no manipulation signals exist, state pricing health clearly.
- End with one actionable fiduciary recommendation: TICK (buy), CLIP (wait), or SKIP (avoid).
- Never fabricate data. Never guess numbers. Never use persuasion language.
- Do NOT use markdown, bullet points, headers, or emojis. Plain paragraph text only."""


async def analyze_manipulation(
    title: str,
    current_price: float | None,
    manipulation_data: dict,
    decision: str,
    hist_low: float | None = None,
    avg_90d: float | None = None,
    market: MarketConfig | None = None,
) -> str:
    """
    Use Claude to write a human-readable analysis of price manipulation findings.

    Returns a plain-text paragraph (3-5 sentences). Returns empty string if
    the API key is missing or the call fails.
    """
    if market is None:
        market = get_market()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ""

    sym = market.currency_symbol

    # Build a data summary for the prompt
    risk = manipulation_data.get("risk_level", "none")
    score = manipulation_data.get("score", 0)
    tactics = manipulation_data.get("tactics", [])
    is_fake = manipulation_data.get("is_fake_deal", False)
    true_market = manipulation_data.get("true_market_price")
    inflated_pct = manipulation_data.get("inflated_by_pct")

    tactics_text = ""
    for t in tactics:
        tactics_text += f"- {t['tactic']} ({t['severity']}): {t['description']}"
        if t.get("evidence"):
            tactics_text += f" Evidence: {t['evidence']}"
        tactics_text += "\n"

    user_msg = f"""Product: {title}
Current Price: {sym}{current_price if current_price else 'N/A'}
Historic Low: {sym}{hist_low if hist_low else 'N/A'}
90-Day Average: {sym}{avg_90d if avg_90d else 'N/A'}
TickClip Decision: {decision}

Manipulation Detection Results:
- Risk Level: {risk}
- Manipulation Score: {score}/100
- Is Fake Deal: {is_fake}
- True Market Price: {sym}{true_market if true_market else 'N/A'}
- Inflated By: {f'{inflated_pct:.0f}%' if inflated_pct else 'N/A'}

Detected Tactics:
{tactics_text if tactics_text else 'None detected.'}

Write your analysis:"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=MANIPULATION_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text.strip()
    except Exception:
        pass

    return ""
