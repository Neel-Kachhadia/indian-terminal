"""Canonical alias registry for NSE index feed symbols.

Single source of truth for which real Angel/NSE instrument-master `name`/
`symbol` strings map to which canonical short index symbol. Every consumer
that needs to resolve an index by name (the DB-backed lookup in
db.write_service's RESOLVE_INDEX_INSTRUMENT, data.instrument_master's
resolve_index_feed(), and live.multi_feed's file-backed token map) must
derive its alias data from this one place -- these string mappings were
previously hand-duplicated at three separate call sites, which is exactly
how a fix or addition (e.g. a missing index) silently failed to reach every
consumer.

Confirmed against the real production instrument master (NSE, AMXIDX rows,
D:\\BharatTerminal\\data\\terminal.db, checked 2026-08-28):

    NIFTY      -> token 99926000, name "NIFTY 50",         symbol "NIFTY"
    BANKNIFTY  -> token 99926009, name "NIFTY BANK",        symbol "BANKNIFTY"
    FINNIFTY   -> token 99926037, name "NIFTY FIN SERVICE", symbol "FINNIFTY"
    MIDCPNIFTY -> token 99926074, name "NIFTY MID SELECT",  symbol "MIDCPNIFTY"
"""

from __future__ import annotations


# canonical_symbol -> set of accepted caller-facing aliases (normalized:
# uppercased, internal whitespace stripped) that resolve to that index.
# Every key AND every value in every alias set is itself a valid lookup key
# (mirrors the pre-existing convention), so callers can pass either the
# short form ("NIFTY") or the space-stripped long form ("NIFTY50").
INDEX_FEED_ALIASES: dict[str, frozenset[str]] = {
    "NIFTY": frozenset({"NIFTY", "NIFTY50"}),
    "NIFTY50": frozenset({"NIFTY", "NIFTY50"}),
    "BANKNIFTY": frozenset({"BANKNIFTY", "NIFTYBANK"}),
    "NIFTYBANK": frozenset({"BANKNIFTY", "NIFTYBANK"}),
    "FINNIFTY": frozenset({"FINNIFTY", "NIFTYFINSERVICE"}),
    "NIFTYFINSERVICE": frozenset({"FINNIFTY", "NIFTYFINSERVICE"}),
    "MIDCPNIFTY": frozenset({"MIDCPNIFTY", "NIFTYMIDSELECT"}),
    "NIFTYMIDSELECT": frozenset({"MIDCPNIFTY", "NIFTYMIDSELECT"}),
}

# canonical_symbol -> the real instrument-master `name` field's literal text
# (with spaces, as Angel actually stores it) -- for consumers that match
# against the raw name string rather than a normalized alias set.
INDEX_CANONICAL_NAME: dict[str, str] = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MID SELECT",
}

# The stable set of canonical short symbols this registry supports (the
# frozenset dict keys above include both short and space-stripped-long
# forms; this is just the short/primary ones, for iteration).
CANONICAL_INDEX_SYMBOLS: tuple[str, ...] = (
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
)
