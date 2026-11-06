"""Search grammar conformance (spec §SEARCH-grammar canonical examples)."""

from __future__ import annotations

import pytest

from glassbox.search.query import _expand_range_token, parse_query

CANONICAL = [
    'FIND RUNS WHERE checkpoint="sha256:abc..." AND N-24-0001 AT token=17 > 0.8',
    'COMPARE_PLACEHOLDER good:tag="good" LIMIT 10000 vs bad:tag="bad" LIMIT 10000',
    "WITH STATS N-24-0001, A-24-07 AT tokens 15..20 AGG mean, p95, firing_rate(0.8)",
    'FIND RUNS WHERE prompt CONTAINS "instructions for" AND N-24-1000 AT token 0..10 MAX >1.2',
    'FIND RUNS WHERE tag="pilot" AND LOGITS AT token=17 > 3.5',
    'FIND RUNS WHERE flag="bad_output" AND A-17-04 AT tokens 10..50 MAX >= 0.9',
]


def test_canonical_find_queries_parse() -> None:
    """All FIND canonicals parse (COMPARE/WITH-STATS use CLI flags, not strings)."""
    find_only = [q for q in CANONICAL if q.startswith("FIND")]
    assert len(find_only) == 4
    parsed = parse_query(find_only[0])
    assert parsed["hot"] == [{"kind": "checkpoint", "value": "abc..."}]
    assert parsed["neuron"][0]["addresses"] == ["N-24-0001"]
    assert parsed["neuron"][0]["token"] == 17

    parsed = parse_query(find_only[1])
    assert parsed["hot"] == [{"kind": "prompt", "value": "instructions for"}]
    neuron = parsed["neuron"][0]
    assert neuron["addresses"] == ["N-24-1000"]
    assert (neuron["lo"], neuron["hi"], neuron["agg"]) == (0, 10, "MAX")

    parsed = parse_query(find_only[2])
    assert parsed["hot"] == [{"kind": "tag", "value": "pilot"}]
    assert parsed["neuron"][0]["addresses"] == ["LOGITS"]

    parsed = parse_query(find_only[3])
    assert parsed["hot"] == [{"kind": "flag", "value": "bad_output"}]
    neuron = parsed["neuron"][0]
    assert (neuron["lo"], neuron["hi"]) == (10, 50)


def test_bare_conditions_without_find_prefix() -> None:
    """CLI passes bare conditions; prefix is optional."""
    parsed = parse_query("N-24-0001 AT token=17 > 0.8")
    assert parsed["neuron"][0]["token"] == 17
    assert parsed["neuron"][0]["op"] == ">"


def test_all_comparators_and_aggs() -> None:
    for op in (">", ">=", "<", "<=", "==", "=", "!="):
        parsed = parse_query(f"N-01-0001 AT token=0 {op} 0.5")
        assert parsed["neuron"][0]["op"] == op
    for agg in ("MAX", "MIN", "MEAN", "P95", "max_window"):
        parsed = parse_query(f"A-01-01 AT tokens 0..9 {agg} > 0.1")
        assert parsed["neuron"][0]["agg"] == ("MAX" if agg == "max_window" else agg)


def test_range_expansion_any_match_documented() -> None:
    assert _expand_range_token("N-24-0001..0003") == ["N-24-0001", "N-24-0002", "N-24-0003"]
    assert _expand_range_token("A-02-01") == ["A-02-01"]
    with pytest.raises(ValueError, match="E02"):
        _expand_range_token("N-99-9999..0001")


def test_bad_grammar_e02() -> None:
    for bad in (
        "",
        "FIND RUNS WHERE",
        "N-24-0001 AT banana > 0.5",
        "hello world",
        "N-24-0001 AT token=5 ~~ 0.5",
        "N-24-0001 AT tokens 9..3 MAX > 0.1",
    ):
        with pytest.raises(ValueError, match="E02"):
            parse_query(bad)


def test_malformed_address_e04_shape() -> None:
    """Malformed addresses fail (E04 at the query layer, E04 message at CLI)."""
    with pytest.raises(ValueError, match="E0"):
        parse_query("NOPE-1 AT token=0 > 0.5")
