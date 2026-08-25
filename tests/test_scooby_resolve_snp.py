from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest

try:
    import embpy.tl.genomics as _real_genomics  # noqa: F401

    _HAS_REAL_EMBPY = True
except ImportError:
    _HAS_REAL_EMBPY = False


if not _HAS_REAL_EMBPY:
    @dataclass
    class _SNPContext:
        position: int
        ref_allele: str
        alt_alleles: list
        context_window: int = 512
        chrom: str = ""
        strand: Literal["+", "-"] = "+"
        variant_id: str = ""

        def __post_init__(self) -> None:
            self.ref_allele = self.ref_allele.upper()
            self.alt_alleles = [a.upper() for a in self.alt_alleles]

    def _genomic_to_bin_indices(intervals, window_start, bin_size, profile_offset_bp=0, num_bins=None):
        bins = set()
        for start, end in intervals:
            rel_start = start - window_start - profile_offset_bp
            rel_end = end - window_start - profile_offset_bp
            b0 = int(rel_start // bin_size)
            b1 = -(-int(rel_end) // bin_size)
            for b in range(max(b0, 0), b1):
                if num_bins is None or 0 <= b < num_bins:
                    bins.add(b)
        return np.array(sorted(bins), dtype=int)

    _fake_embpy = types.ModuleType("embpy")
    _fake_embpy_tl = types.ModuleType("embpy.tl")
    _fake_embpy_genomics = types.ModuleType("embpy.tl.genomics")
    _fake_embpy_genomics.SNPContext = _SNPContext
    _fake_embpy_genomics.genomic_to_bin_indices = _genomic_to_bin_indices
    _fake_embpy_tl.genomics = _fake_embpy_genomics
    _fake_embpy.tl = _fake_embpy_tl
    sys.modules.setdefault("embpy", _fake_embpy)
    sys.modules.setdefault("embpy.tl", _fake_embpy_tl)
    sys.modules.setdefault("embpy.tl.genomics", _fake_embpy_genomics)

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


def test_bin_indices_clipped_to_num_bins():
    """Regression test for the unclipped-bin-index bug: exons far outside the model's
    cropped profile region must be dropped, not returned as out-of-range indices that
    would crash downstream indexing."""
    result = resolve_snp_and_exon_bins(
        a0="A", a1="G", exon_intervals=[(POS - 10, POS + 10), (POS + 100_000, POS + 100_020)],
        **{**COMMON_KWARGS, "num_bins": 150},
    )
    assert result is not None
    _, bins = result
    assert len(bins) == 20  # only the near exon's bins survive
    assert bins.max() < 150
