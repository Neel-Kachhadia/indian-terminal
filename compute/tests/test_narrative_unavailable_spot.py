from __future__ import annotations

from intelligence.groq_client import _template_narrative
from intelligence.narrative import _enforce_brief_contract, _has_enough_context


def test_template_narrative_reports_honest_unavailable_state_without_spot() -> None:
    """Reproduces a real defect: the templated market-brief fallback used
    to embed the literal placeholder string "spot unavailable" mid-sentence
    ("NIFTY trades near spot unavailable with a CALM regime...") whenever
    spot was missing, instead of stating plainly that the brief itself is
    unavailable. Observed live in the installed GUI's Groq Brief / header
    brief tooltip."""
    brief = _template_narrative({"spot": None, "regime": "CALM", "direction": "UP"})
    assert "spot unavailable" not in brief
    assert "trades near" not in brief
    assert "unavailable" in brief.lower()


def test_template_narrative_still_builds_full_brief_when_spot_is_real() -> None:
    brief = _template_narrative(
        {
            "spot": 24175.65,
            "gex_flip": 24200,
            "regime": "CALM",
            "direction": "UP",
            "confidence": 48,
            "max_pain": 24000,
        }
    )
    assert "24,176" in brief or "24,175" in brief
    assert "GEX flip" in brief
    assert "spot unavailable" not in brief


def test_enforce_brief_contract_does_not_loop_into_the_broken_placeholder_text() -> None:
    context = {"spot": None}
    assert _has_enough_context(context) is False
    enforced = _enforce_brief_contract(_template_narrative(context), context)
    assert "spot unavailable" not in enforced
    assert "unavailable" in enforced.lower()
