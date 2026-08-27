from __future__ import annotations

import pytest

from cellink.resources._ld import check_ancestry_match


def test_ancestry_match_warns_on_real_east_asian_vs_eur(caplog):
    check_ancestry_match("EUR", ["East Asian"], on_mismatch="warn")
    assert any("Ancestry mismatch" in r.message for r in caplog.records)


def test_ancestry_match_silent_when_population_matches():
    check_ancestry_match("EAS", ["East Asian"], on_mismatch="warn") 


def test_ancestry_match_raises_in_error_mode():
    with pytest.raises(ValueError, match="Ancestry mismatch"):
        check_ancestry_match("EUR", ["East Asian"], on_mismatch="error")


def test_ancestry_match_noop_when_ancestry_unknown():
    check_ancestry_match("EUR", None) 


def test_ancestry_match_ignore_mode_suppresses():
    check_ancestry_match("EUR", ["East Asian"], on_mismatch="ignore")


def test_ancestry_match_accepts_string_or_list():
    with pytest.raises(ValueError):
        check_ancestry_match("EUR", "East Asian", on_mismatch="error")


def test_ancestry_match_unsupported_ancestry_has_no_suggestion(caplog):
    check_ancestry_match("EUR", ["African"], on_mismatch="warn")
    msgs = [r.message for r in caplog.records if "Ancestry mismatch" in r.message]
    assert msgs
    assert "does not ship a matching 1000G panel" in msgs[0]


def test_ancestry_match_handles_real_gwas_catalog_discovery_ancestry_free_text():
    real_value = ["178616 East Asian (Japan)"]
    with pytest.raises(ValueError, match="Ancestry mismatch"):
        check_ancestry_match("EUR", real_value, on_mismatch="error")
    check_ancestry_match("EAS", real_value, on_mismatch="error")  # must not raise: correct match
