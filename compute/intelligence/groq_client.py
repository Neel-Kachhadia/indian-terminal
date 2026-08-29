from __future__ import annotations

from typing import Any

from security.credential_manager import get_groq_api_key


GROQ_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]
def groq_complete(
    prompt: str,
    system: str = "You are a senior quant analyst at a top Indian hedge fund.",
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> tuple[str, str, bool]:
    api_keys = _load_api_keys()
    key = api_keys[0] if api_keys else None
    if not key:
        return _template_narrative(), "template_fallback", False

    try:
        from groq import Groq
    except Exception as error:
        print(f"Groq package import failed: {error!r}")
        return f"{_template_narrative()} Groq package unavailable: {error}", "template_fallback", False

    for key_index, api_key in enumerate(api_keys, start=1):
        try:
            client = Groq(api_key=api_key, timeout=15.0, max_retries=0)
            for model in FALLBACK_MODELS:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    content = response.choices[0].message.content or ""
                    return _sanitize_narrative(content.strip()), model, True
                except Exception as error:
                    print(f"Groq key #{key_index} model {model} failed: {type(error).__name__}")
                    continue
        finally:
            del api_key
    return _template_narrative(), "template_fallback", False


def _load_api_key() -> str | None:
    keys = _load_api_keys()
    return keys[0] if keys else None


def _load_api_keys() -> list[str]:
    try:
        key = get_groq_api_key()
    except Exception:
        key = None
    return [key] if key else []


def _template_narrative(context: dict[str, Any] | None = None) -> str:
    ctx = context or {}
    if ctx.get("spot") is None:
        # Spot is the anchor of the very first sentence -- without it the
        # rest of the templated narrative can only be patched together with
        # placeholder tokens like "spot unavailable" embedded mid-sentence
        # ("NIFTY trades near spot unavailable with a..."), which reads as a
        # leaked error rather than an honest unavailable state. Say so
        # directly instead of fabricating a narrative around missing data.
        return "Market brief unavailable: live spot price has not been established yet."
    spot = _format_price(ctx.get("spot")) or "spot unavailable"
    flip = _format_price(ctx.get("gex_flip")) or "GEX flip unavailable"
    regime = ctx.get("regime") or "range-bound"
    direction = ctx.get("direction") or "FLAT"
    confidence = ctx.get("confidence") or 0
    vix = _format_number(ctx.get("vix"), 2) or "unavailable"
    max_pain = _format_price(ctx.get("max_pain")) or "max pain unavailable"
    pcr = _format_number(ctx.get("pcr"), 2) or "unavailable"
    fii_net = _format_number(ctx.get("fii_net"), 0) or "unavailable"
    return (
        f"NIFTY trades near {spot} with a {regime} regime and {direction} ML bias "
        f"at {confidence}% confidence. Options structure keeps the GEX flip level "
        f"at {flip} as the key line: holding near it favours mean reversion, while "
        f"a clean move away from it can let intraday momentum expand. Max pain is "
        f"near {max_pain}, so expiry gravity remains relevant if spot stays within "
        f"one normal session range. India VIX is {vix}, PCR is {pcr}, and FII net "
        f"cash flow is ₹{fii_net}Cr, keeping position sizing sensitive to "
        f"volatility shocks rather than direction alone. Conformal "
        f"intervals and Bayesian uncertainty should be treated as the guardrails: "
        f"when width expands, reduce conviction even if headline direction improves. "
        f"Pattern scores, options flow, and VaR should agree before exposure rises; "
        f"if they diverge, the cleaner decision is patience and smaller notional risk. "
        f"Watch the GEX flip level by name and the first sustained break beyond the "
        f"range around it; that is the clearest tell for whether compression persists "
        f"or volatility starts to expand."
    )


def _sanitize_narrative(text: str) -> str:
    cleaned = text.replace(" buy ", " consider ").replace(" sell ", " consider ")
    cleaned = cleaned.replace("Buy ", "Consider ").replace("Sell ", "Consider ")
    return cleaned


def _format_price(value: Any) -> str | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return f"₹{number:,.0f}"


def _format_number(value: Any, digits: int) -> str | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return f"{number:,.{digits}f}"


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number
