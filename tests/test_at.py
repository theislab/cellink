import logging

import numpy as np
import pytest
from anndata import AnnData

from cellink._core.dummy_data import sim_gdata
from cellink.at import utils
from cellink.at.acat import compute_acat
from cellink.at.gwas import GWAS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_generate_phenotype_data():
    """Test the generate_phenotype_data function."""
    # Generate phenotype data
    N = 10000  # number of individuals
    S = 100  # number of variants
    number_causal_variants = 10  # number of causal variants
    vg = 0.5  # variance explained by the causal variants

    X = np.random.choice([0, 1, 2], size=(N, S))
    X = np.asarray(X, dtype=np.float64)

    Y, betas, Yg, Yn = utils.generate_phenotype(X, vg=vg, number_causal_variants=number_causal_variants)

    assert Y.shape == (N, 1), "Generated phenotype data shape is incorrect"

    # Test the mean and standard deviation of the generated phenotype data
    np.testing.assert_allclose(
        Y.mean(axis=0),
        0,
        rtol=1e-1,
        atol=1e-1,
        err_msg="Generated phenotype data mean is incorrect",
    )

    np.testing.assert_allclose(
        Y.std(axis=0),
        1,
        rtol=1e-1,
        atol=1e-1,
        err_msg="Generated phenotype data std is incorrect",
    )

    assert betas.shape == (S, 1), "Generated beta data shape is incorrect"

    np.testing.assert_allclose(
        Yg.var(0),
        vg,
        rtol=1e-1,
        atol=1e-1,
        err_msg="Generated Yg data variance is incorrect",
    )

    np.testing.assert_allclose(
        Yg.mean(0),
        0,
        rtol=1e-1,
        atol=1e-1,
        err_msg="Generated Yg data mean is incorrect",
    )

    np.testing.assert_allclose(
        Yn.var(0),
        1 - vg,
        rtol=1e-1,
        atol=1e-1,
        err_msg="Generated Yn data variance is incorrect",
    )

    np.testing.assert_allclose(
        Yn.mean(0),
        0,
        rtol=1e-1,
        atol=1e-1,
        err_msg="Generated Yn data mean is incorrect",
    )


def test_burden_testing():
    """Test the burden testing function."""
    np.random.seed(0)  # noisy p-value comparison, pin the seed so this isn't flaky
    N = 100000  # number of individuals
    S = 300  # number of variants
    number_causal_variants = 50  # number of causal variants
    vg = 0.3  # variance explained by the causal variants

    gdata = sim_gdata(n_donors=N, n_snps=S)
    gdata.X = np.asarray(np.random.choice([0, 1, 2], size=(N, S), p=[0.99, 0.005, 0.005]), dtype=np.float64)

    Y, *_ = utils.generate_phenotype(gdata.X, vg=vg, number_causal_variants=number_causal_variants)
    gdata.obs["pheno"] = Y.ravel()
    gdata.obs["pheno_perm"] = np.random.permutation(Y.ravel())

    # the burden is the sum of the variants (arbitrary); a derived vector has to be
    # wrapped in its own object, since the models only take DonorData/AnnData now
    burden = AnnData(X=gdata.X.sum(axis=1, keepdims=True), obs=gdata.obs)

    gwas = GWAS(Y="pheno", data=gdata)
    gwas.test_association(burden)
    pv = gwas.getPv()
    assert pv is not None, "P-value is None"

    gwas = GWAS(Y="pheno_perm", data=gdata)
    gwas.test_association(burden)
    pvp = gwas.getPv()  # pv permutated

    assert pv < pvp, "P-value is not smaller than permutated p-value"


def test_skat_testing():
    pytest.importorskip("chiscore", reason="Skat needs chiscore, install with `conda install -c conda-forge chiscore`")
    from cellink.at.skat import Skat

    N = 10000  # number of individuals
    S = 10  # number of variants
    number_causal_variants = 3  # number of causal variants
    vg = 0.1  # variance explained by the causal variants

    gdata = sim_gdata(n_donors=N, n_snps=S)
    gdata.X = np.asarray(np.random.choice([0, 1, 2], size=(N, S), p=[0.9, 0.05, 0.05]), dtype=np.float64)
    logger.info(f"number of variants: {gdata.X.sum(1)}")

    Y, *_ = utils.generate_phenotype(gdata.X, vg=vg, number_causal_variants=number_causal_variants)
    gdata.obs["pheno"] = Y.ravel()
    gdata.obs["pheno_perm"] = np.random.permutation(Y.ravel())

    skat = Skat(min_threshold=10)
    pv = skat.run_test(Y="pheno", data=gdata)  # variants come from gdata.X
    assert pv is not None, "P-value is None"

    pvp = skat.run_test(Y="pheno_perm", data=gdata)  # pv permutated
    logger.info(f"pv: {pv}, pvp: {pvp}")

    assert pv < pvp, "P-value is not smaller than permutated p-value"


def test_acat_testing():
    pytest.importorskip("chiscore", reason="Skat needs chiscore, install with `conda install -c conda-forge chiscore`")
    from cellink.at.skat import Skat

    N = 10000  # number of individuals
    S = 300  # number of variants
    number_causal_variants = 50  # number of causal variants
    vg = 0.15  # variance explained by the causal variants

    gdata = sim_gdata(n_donors=N, n_snps=S)
    gdata.X = np.asarray(np.random.choice([0, 1, 2], size=(N, S), p=[0.9, 0.05, 0.05]), dtype=np.float64)

    Y, *_ = utils.generate_phenotype(gdata.X, vg=vg, number_causal_variants=number_causal_variants)
    gdata.obs["pheno"] = Y.ravel()
    gdata.obs["pheno_perm"] = np.random.permutation(Y.ravel())

    skat = Skat(min_threshold=10)
    skat_pv = skat.run_test(Y="pheno", data=gdata)  # variants come from gdata.X
    skat_pvp = skat.run_test(Y="pheno_perm", data=gdata)  # pv permutated

    # the burden here is the sum of the standardized variants (arbitrary)
    X_std = (gdata.X - gdata.X.mean(0)) / utils.xgower_factor_(gdata.X)
    burden = AnnData(X=X_std.sum(axis=1, keepdims=True), obs=gdata.obs)

    gwas = GWAS(Y="pheno", data=gdata)
    gwas.test_association(burden)
    burden_pv = gwas.getPv()

    gwas = GWAS(Y="pheno_perm", data=gdata)
    gwas.test_association(burden)
    burden_pvp = gwas.getPv()  # pv permutated

    # ACAT testing
    pvs = np.stack([skat_pv, burden_pv], axis=1)
    pvs = utils.ensure_float64_array(pvs)
    pvs = pvs.reshape(1, -1)
    acat_pv = compute_acat(pvs=pvs)
    pvps = np.stack([skat_pvp, burden_pvp], axis=1)
    pvps = utils.ensure_float64_array(pvps)
    pvps = pvps.reshape(1, -1)
    acat_pvp = compute_acat(pvs=pvps)
    logger.info(f"acat_pv: {acat_pv}, acat_pvp: {acat_pvp}")
    assert acat_pv < acat_pvp, "P-value is not smaller than permutated p-value"


def _simulate_for_skat(seed, ve_g, ve_cov=0.3, confounded=False, N=500, S=30, K=8):
    """Genotypes plus a phenotype built from an explicit variance budget.

    The causal effects have balanced signs (``sum(betas) == 0``), so the burden of the
    set carries no signal and only the variance component does -- which is what SKAT
    tests. The phenotype is rank-inverse-normal transformed, as a QTL pipeline would.

    With ``confounded=True`` the covariate is correlated with the genetic component and
    the phenotype has no genetic term of its own (``ve_g=0``): the genotypes reach the
    phenotype only through the covariate.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    unit = lambda x: (x - x.mean()) / x.std()

    gdata = sim_gdata(n_donors=N, n_snps=S)
    gdata.X = np.asarray(rng.choice([0, 1, 2], size=(N, S), p=[0.9, 0.05, 0.05]), dtype=np.float64)

    betas = np.zeros(S)
    idx = rng.choice(S, K, replace=False)
    betas[idx[: K // 2]], betas[idx[K // 2 :]] = 1.0, -1.0
    Yg = unit((gdata.X - gdata.X.mean(0)) @ betas)

    cov = unit(Yg + rng.standard_normal(N)) if confounded else rng.standard_normal(N)
    y = np.sqrt(ve_g) * Yg + np.sqrt(ve_cov) * cov + np.sqrt(1 - ve_g - ve_cov) * rng.standard_normal(N)

    gdata.obs["pheno"] = stats.norm.ppf((stats.rankdata(y) - 0.5) / N)
    gdata.obs["cov"] = cov
    return gdata


def test_skat_covariate_removes_confounding():
    """A covariate correlated with the genotypes creates a false positive; adjusting for it removes it.

    The phenotype has no genetic component at all (ve_g = 0), so any association is
    confounding by construction. Omitting the covariate must find it; passing it as `F`
    must not.
    """
    pytest.importorskip("chiscore", reason="Skat needs chiscore, install with `conda install -c conda-forge chiscore`")
    from cellink.at.skat import Skat

    skat = Skat(min_threshold=10)
    adjusted = []
    for seed in range(5):
        gdata = _simulate_for_skat(seed, ve_g=0.0, confounded=True)
        unadj = float(np.ravel(skat.run_test(Y="pheno", data=gdata))[0])
        adj = float(np.ravel(skat.run_test(Y="pheno", F="cov", data=gdata))[0])

        assert unadj < 1e-4, f"seed {seed}: confounding should be detected, got {unadj}"
        # The ratio, not an absolute threshold: the adjusted p-value is a draw from a
        # correctly calibrated null, so it sits below 0.05 about 5% of the time.
        assert adj > 1e3 * unadj, f"seed {seed}: adjusting barely moved the p-value ({unadj} -> {adj})"
        adjusted.append(adj)

    assert np.median(adjusted) > 0.05, f"adjusted p-values should be null-like, got {adjusted}"


def test_skat_covariate_improves_power():
    """Adjusting for an independent covariate removes noise, so a real signal gets easier to see."""
    pytest.importorskip("chiscore", reason="Skat needs chiscore, install with `conda install -c conda-forge chiscore`")
    from cellink.at.skat import Skat

    skat = Skat(min_threshold=10)
    for seed in range(5):
        gdata = _simulate_for_skat(seed, ve_g=0.2, confounded=False)
        unadj = float(np.ravel(skat.run_test(Y="pheno", data=gdata))[0])
        adj = float(np.ravel(skat.run_test(Y="pheno", F="cov", data=gdata))[0])

        assert adj < unadj, f"seed {seed}: adjusting should sharpen the signal ({unadj} -> {adj})"
        assert adj < 1e-8, f"seed {seed}: planted ve_g=0.2 should be strongly detected, got {adj}"
