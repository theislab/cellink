from __future__ import annotations

import numpy as np
import pytest

from cellink.tl import coloc_abf, coloc_susie
from cellink.tl._coloc import DEFAULT_PRIOR_VAR, _combine_log_abf, _log_abf

rpy2 = pytest.importorskip("rpy2", reason="rpy2 not installed")
robjects = pytest.importorskip("rpy2.robjects", reason="rpy2 not installed")
from rpy2.robjects.packages import PackageNotInstalledError, importr  # noqa: E402

try:
    coloc_r = importr("coloc")
except PackageNotInstalledError:
    pytest.skip("R package 'coloc' not installed", allow_module_level=True)


def _r_log_abf(beta: np.ndarray, se: np.ndarray, prior_var: float) -> np.ndarray:
    """R coloc's own `approx.bf.estimates(z, V, type="quant", sdY=sqrt(prior_var)/0.15)`,
    called with sdY chosen so that R's internal `sd.prior = 0.15 * sdY` equals
    `sqrt(prior_var)` exactly, matching cellink's `prior_var` parameterization."""
    z = np.asarray(beta) / np.asarray(se)
    v = np.asarray(se) ** 2
    sdy = float(np.sqrt(prior_var) / 0.15)
    out = coloc_r.approx_bf_estimates(
        z=robjects.FloatVector(z), V=robjects.FloatVector(v), type="quant", sdY=robjects.FloatVector([sdy])
    )
    labf_col = list(out.names).index("lABF")
    return np.asarray(out[labf_col])


def _r_combine_abf(l1: np.ndarray, l2: np.ndarray, p1: float, p2: float, p12: float) -> dict[str, float]:
    out = coloc_r.combine_abf(
        robjects.FloatVector(np.asarray(l1)),
        robjects.FloatVector(np.asarray(l2)),
        robjects.FloatVector([p1]),
        robjects.FloatVector([p2]),
        robjects.FloatVector([p12]),
        quiet=True,
    )
    # R names its output PP.H0.abf..PP.H4.abf; remap to PP0..PP4 to match cellink's keys
    r_to_py_key = {f"PP.H{i}.abf": f"PP{i}" for i in range(5)}
    return {r_to_py_key[name]: float(out.rx2(name)[0]) for name in out.names}


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_log_abf_matches_r_coloc(rng):
    n = 200
    beta = rng.normal(0, 0.03, n)
    se = rng.uniform(0.01, 0.06, n)
    py_labf = _log_abf(beta, se, DEFAULT_PRIOR_VAR)
    r_labf = _r_log_abf(beta, se, DEFAULT_PRIOR_VAR)
    np.testing.assert_allclose(py_labf, r_labf, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("p1,p2,p12", [(1e-4, 1e-4, 1e-5), (5e-5, 2e-4, 1e-6)])
def test_combine_log_abf_matches_r_coloc(rng, p1, p2, p12):
    n = 150
    l1 = rng.normal(0, 3, n)
    l2 = rng.normal(0, 3, n)
    # give a handful of SNPs a real shared/distinct signal so all 5 hypotheses
    # get non-negligible mass, not just a diffuse null
    l1[10] += 15
    l2[10] += 12
    l1[50] += 20
    l2[90] += 18

    py_pp = _combine_log_abf(l1, l2, p1, p2, p12)
    r_pp = _r_combine_abf(l1, l2, p1, p2, p12)

    assert set(py_pp) == {"PP0", "PP1", "PP2", "PP3", "PP4"}
    for key in py_pp:
        assert py_pp[key] == pytest.approx(r_pp[key], abs=1e-8), f"{key}: py={py_pp[key]!r} r={r_pp[key]!r}"
    assert sum(py_pp.values()) == pytest.approx(1.0, abs=1e-10)


def test_coloc_abf_end_to_end_matches_r_coloc(rng):
    """Full coloc_abf pipeline (beta/se -> log-ABF -> H0-H4), not just its
    two halves tested in isolation above."""
    n = 100
    beta1 = rng.normal(0, 0.02, n)
    se1 = rng.uniform(0.015, 0.05, n)
    beta2 = rng.normal(0, 0.02, n)
    se2 = rng.uniform(0.015, 0.05, n)
    # SNP 7: a real shared causal signal, strong in both studies
    beta1[7], se1[7] = 0.6, 0.04
    beta2[7], se2[7] = 0.5, 0.05

    py_pp = coloc_abf(beta1, se1, beta2, se2)

    l1 = _r_log_abf(beta1, se1, DEFAULT_PRIOR_VAR)
    l2 = _r_log_abf(beta2, se2, DEFAULT_PRIOR_VAR)
    r_pp = _r_combine_abf(l1, l2, p1=1e-4, p2=1e-4, p12=1e-5)

    for key in py_pp:
        assert py_pp[key] == pytest.approx(r_pp[key], abs=1e-6)
    assert py_pp["PP4"] > 0.9  # sanity: the planted shared signal should dominate


def test_coloc_susie_pairwise_combination_matches_r_coloc(rng):
    """coloc_susie delegates each (signal, signal) pair to the same
    _combine_log_abf core coloc_abf uses; check that per-pair result
    against R directly rather than trusting the delegation blindly."""
    n = 40
    lbf1 = rng.normal(0, 1, (2, n))
    lbf2 = rng.normal(0, 1, (2, n))
    lbf1[0, 15] = 25.0  # effect 0 in study 1 points at SNP 15
    lbf2[0, 15] = 22.0  # effect 0 in study 2 also points at SNP 15: real colocalization
    lbf1[1, 30] = 20.0  # effect 1 in study 1 points at SNP 30, no match in study 2

    res = coloc_susie(lbf1, lbf2, cs1_index=[0, 1], cs2_index=[0])

    for _, row in res.iterrows():
        l1 = lbf1[int(row["idx1"])]
        l2 = lbf2[int(row["idx2"])]
        r_pp = _r_combine_abf(l1, l2, p1=1e-4, p2=1e-4, p12=1e-5)
        for key in ["PP0", "PP1", "PP2", "PP3", "PP4"]:
            assert row[key] == pytest.approx(r_pp[key], abs=1e-6), f"idx1={row['idx1']} idx2={row['idx2']} {key}"

    # the matched pair (0, 0) should show real colocalization; the unmatched
    # pair (1, 0) should not
    hit00 = res[(res["idx1"] == 0) & (res["idx2"] == 0)].iloc[0]
    hit10 = res[(res["idx1"] == 1) & (res["idx2"] == 0)].iloc[0]
    assert hit00["PP4"] > 0.9
    assert hit10["PP4"] < hit00["PP4"]
