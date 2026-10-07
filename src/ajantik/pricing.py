"""Model prices, USD per million tokens.

Source: Anthropic model table bundled with the claude-api skill (cached 2026-06-24).
Cache writes (5-minute TTL) cost 1.25x input, cache reads 0.1x input (same source,
prompt-caching guide). Re-verify before quoting prices to anyone.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_WRITE = 1.25
CACHE_READ = 0.10
PRICE_SOURCE = "claude-api skill model table, cached 2026-06-24"


def cost_usd(model: str, usage: dict[str, int]) -> float:
    if model not in PRICES:
        raise KeyError(f"No price known for model: {model}. Add a sourced price to pricing.py.")
    inp, out = PRICES[model]
    return (
        usage.get("input_tokens", 0) * inp
        + usage.get("cache_creation_input_tokens", 0) * inp * CACHE_WRITE
        + usage.get("cache_read_input_tokens", 0) * inp * CACHE_READ
        + usage.get("output_tokens", 0) * out
    ) / 1_000_000


def worst_case_call_usd(model: str, prompt_tokens: int, max_tokens: int) -> float:
    """Upper bound for one call: whole prompt uncached, output hits max_tokens."""
    return cost_usd(model, {"input_tokens": prompt_tokens, "output_tokens": max_tokens})
