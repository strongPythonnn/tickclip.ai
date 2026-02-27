"""
TickClip.ai — Market configuration for multi-region support (US, UK).

Each MarketConfig defines Amazon domains, currency, retailers, and search
parameters for a specific market.  Resolved per-request from the frontend's
detected locale.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarketConfig:
    code: str
    keepa_domain: int
    amazon_host: str
    amazon_marketplace: str
    amazon_tld: str
    amazon_region: str
    image_cdn: str
    currency_code: str
    currency_symbol: str
    serper_gl: str
    camel_base: str
    retailer_queries: dict = field(default_factory=dict)
    retailer_meta: dict = field(default_factory=dict)
    deal_queries: dict = field(default_factory=dict)
    price_regex: str = r"\$[\d,]+\.?\d*"


_MARKETS: dict[str, MarketConfig] = {
    "us": MarketConfig(
        code="us",
        keepa_domain=1,
        amazon_host="webservices.amazon.com",
        amazon_marketplace="www.amazon.com",
        amazon_tld="amazon.com",
        amazon_region="us-east-1",
        image_cdn="images-na",
        currency_code="USD",
        currency_symbol="$",
        serper_gl="us",
        camel_base="camelcamelcamel.com",
        retailer_queries={
            "walmart": "site:walmart.com {product} buy",
            "bestbuy": "site:bestbuy.com {product}",
            "target": "site:target.com {product}",
            "costco": "site:costco.com {product}",
            "newegg": "site:newegg.com {product}",
        },
        retailer_meta={
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
        },
        deal_queries={
            "slickdeals": "site:slickdeals.net {product} deal",
            "retailmenot": "site:retailmenot.com {product} coupon",
            "groupon": "site:groupon.com {product}",
            "camelcamelcamel": "site:camelcamelcamel.com {product}",
            "honey": "site:joinhoney.com {product}",
        },
        price_regex=r"\$[\d,]+\.?\d*",
    ),
    "uk": MarketConfig(
        code="uk",
        keepa_domain=2,
        amazon_host="webservices.amazon.co.uk",
        amazon_marketplace="www.amazon.co.uk",
        amazon_tld="amazon.co.uk",
        amazon_region="eu-west-1",
        image_cdn="images-eu",
        currency_code="GBP",
        currency_symbol="\u00a3",
        serper_gl="uk",
        camel_base="uk.camelcamelcamel.com",
        retailer_queries={
            "argos": "site:argos.co.uk {product}",
            "currys": "site:currys.co.uk {product}",
            "johnlewis": "site:johnlewis.com {product}",
            "ao": "site:ao.com {product}",
            "very": "site:very.co.uk {product}",
        },
        retailer_meta={
            "argos.co.uk": {"name": "Argos", "color": "#d82730"},
            "currys.co.uk": {"name": "Currys", "color": "#1c1f72"},
            "johnlewis.com": {"name": "John Lewis", "color": "#002f35"},
            "ao.com": {"name": "AO", "color": "#1f69ad"},
            "very.co.uk": {"name": "Very", "color": "#9c1e84"},
            "amazon.co.uk": {"name": "Amazon UK", "color": "#ff9900"},
            "hotukdeals.com": {"name": "HotUKDeals", "color": "#0081d5"},
            "camelcamelcamel.com": {"name": "CamelCamelCamel", "color": "#884499"},
            "vouchercodes.co.uk": {"name": "VoucherCodes", "color": "#ff5a00"},
            "ebay.co.uk": {"name": "eBay UK", "color": "#e53238"},
            "moneysavingexpert.com": {"name": "MoneySavingExpert", "color": "#004b8d"},
        },
        deal_queries={
            "hotukdeals": "site:hotukdeals.com {product} deal",
            "vouchercodes": "site:vouchercodes.co.uk {product} code",
            "camelcamelcamel": "site:camelcamelcamel.com {product}",
            "moneysavingexpert": "site:moneysavingexpert.com {product}",
        },
        price_regex=r"(?:\u00a3|GBP\s?)[\d,]+\.?\d*",
    ),
}

DEFAULT_MARKET = os.getenv("TICKCLIP_DEFAULT_MARKET", "us").lower()


def get_market(code: str | None = None) -> MarketConfig:
    """Resolve a MarketConfig by code.  Falls back to default, then US."""
    if code:
        code = code.lower().strip()
    if not code or code not in _MARKETS:
        code = DEFAULT_MARKET if DEFAULT_MARKET in _MARKETS else "us"
    return _MARKETS[code]
