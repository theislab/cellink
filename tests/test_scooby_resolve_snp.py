from __future__ import annotations

import pytest

pytest.importorskip("embpy", reason="resolve_snp_and_exon_bins needs embpy, install with `pip install cellink[embpy]`")

from cellink.tl.external import resolve_snp_and_exon_bins  # noqa: E402

WINDOW = "N" * 100 + "A" + "N" * 100  # actual reference base at snp_offset=101 (1-based) is "A"
SNP_OFFSET = 101
POS = 1000
WINDOW_START = POS - SNP_OFFSET
COMMON_KWARGS = dict(
    chrom="chr1",
    pos=POS,
    window=WINDOW,
    snp_offset=SNP_OFFSET,
    window_start=WINDOW_START,
    bin_size=1,
    profile_offset_bp=0,
    num_bins=10000,
    context_window=201,
)


def test_no_swap_when_a0_matches_reference():
    result = resolve_snp_and_exon_bins(a0="A", a1="G", exon_intervals=[(POS - 10, POS + 10)], **COMMON_KWARGS)
    assert result is not None
    snp, bins = result
    assert snp.ref_allele == "A"
    assert snp.alt_alleles == ["G"]
    assert len(bins) == 20


def test_swap_when_a1_matches_reference():
    """Regression test for the ref/alt allele-order bug: a0/a1 aren't guaranteed to be in
    forward-strand reference/alternate order, so scoring must check the real reference
    base rather than trusting a0 is always the reference."""
    result = resolve_snp_and_exon_bins(a0="G", a1="A", exon_intervals=[(POS - 10, POS + 10)], **COMMON_KWARGS)
    assert result is not None
    snp, _ = result
    assert snp.ref_allele == "A"
    assert snp.alt_alleles == ["G"]


def test_skips_when_neither_allele_matches_reference():
    result = resolve_snp_and_exon_bins(a0="C", a1="T", exon_intervals=[(POS - 10, POS + 10)], **COMMON_KWARGS)
    assert result is None


def test_skips_when_no_exon_bins_overlap():
    """Regression test for the silent-no-op bug: with no exon bins in the scored window,
    there is nothing to score, and this must be reported as a skip, not scored as a
    spurious zero effect."""
    result = resolve_snp_and_exon_bins(a0="A", a1="G", exon_intervals=[], **COMMON_KWARGS)
    assert result is None
