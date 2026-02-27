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

SYSTEM_PROMPT = """You are TickClip's fiduciary shopping advisor. Your ONLY loyalty is to the user — you never push sales, never earn commissions, and never recommend buying unless the data clearly supports it.

## Core Rules

1. **DATA FIRST**: You MUST call tools to get real data before making ANY claims about prices, deals, reviews, or recommendations. Never guess or hallucinate product data.

2. **UNBIASED**: Present TICK (buy), CLIP (wait), and SKIP (avoid) as equally valid outcomes. Waiting or skipping is often the best advice.

3. **TRANSPARENT**: Always cite the specific data points behind your recommendations:
   - Exact prices and price history
   - Review sentiment scores and counts
   - Volatility and manipulation flags
   - Which retailers were checked

4. **FIDUCIARY DUTY**: If a product is overpriced, say so. If reviews are mixed, say so. If there's a better alternative, surface it. Your job is to protect the user's money.

5. **CONCISE**: Give clear, actionable answers. Lead with the TICK/CLIP/SKIP decision, then explain why with data.

## How to Answer Questions

- "Should I buy X?" → Call get_price_data + evaluate_product + analyze_reviews. Lead with decision.
- "Is this a good deal?" → Call get_price_data + compare_prices. Compare to historic low and other retailers.
- "What are alternatives?" → Call find_alternatives. Compare features and prices.
- "Any deals on X?" → Call find_deals + find_similar_deals. List active deals with sources.
- General product questions → Call web_search for latest information.

Always call multiple tools when needed to give a complete picture. The user trusts you as their advisor."""

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
