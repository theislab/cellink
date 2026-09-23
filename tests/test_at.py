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


def test_run_burden_test_resolves_from_anndata():
    """`tl.run_burden_test` builds its GWAS from the genotype object and returns one row per annotation."""
    import pandas as pd

    from cellink.tl import run_burden_test

    rng = np.random.default_rng(3)
    N, S = 300, 20
    gdata = sim_gdata(n_donors=N, n_snps=S)
    gdata.X = np.asarray(rng.choice([0, 1, 2], size=(N, S), p=[0.9, 0.05, 0.05]), dtype=np.float64)

    annotation_cols = ["maf_beta", "tss_distance"]
    gdata.varm["variant_annotation"] = pd.DataFrame(
        rng.random((S, len(annotation_cols))), index=gdata.var_names, columns=annotation_cols
    )

    # phenotype and covariates live on the object the burdens are computed from
    gdata.obs["pheno"] = rng.standard_normal(N)
    gdata.obs["age"] = rng.standard_normal(N)

    rdf = run_burden_test(gdata, "pheno", "age", gene="GENE1", annotation_cols=annotation_cols)

    assert list(rdf["weight_col"]) == annotation_cols
    assert len(rdf) == len(annotation_cols)
    assert np.all(np.isfinite(rdf["pv"])) and np.all((rdf["pv"] >= 0) & (rdf["pv"] <= 1))
    assert set(rdf.columns) >= {"burden_gene", "egene", "weight_col", "pv", "beta", "betaste", "lrt"}


def _structlmm_donordata(seed=0, n_donors=40, n_snps=4):
    """A DonorData with a cell-level phenotype and a cell-state factor to use as E."""
    from cellink import DonorData
    from cellink._core.dummy_data import sim_adata

    rng = np.random.default_rng(seed)
    dd = DonorData(G=sim_gdata(n_donors=n_donors, n_snps=n_snps), C=sim_adata(n_donors=n_donors))
    dd.G.obs["sex"] = rng.integers(0, 2, dd.G.n_obs).astype(float)
    dd.C.obs["expr"] = rng.standard_normal(dd.C.n_obs)
    # `sim_adata` stores celltype as object dtype, so there are no empty levels here;
    # with a real Categorical, call `.cat.remove_unused_categories()` first or the
    # one-hot gains all-zero columns and E becomes singular
    return dd


def test_structlmm_cell_level_resolves_e_from_formula():
    """E is a formula like any other slot: a one-hot of the cell state, resolved at cell level."""
    pytest.importorskip("limix_core", reason="StructLMM needs limix-core")
    pytest.importorskip("chiscore", reason="StructLMM needs chiscore")
    from cellink.at.structlmm import StructLMM

    dd = _structlmm_donordata()
    slmm = StructLMM(y="expr", E="celltype - 1", F="crepeat(sex)", data=dd, target_level="cell")

    assert slmm.y.shape == (dd.C.n_obs, 1)
    assert slmm.E.shape[0] == dd.C.n_obs and slmm.E.shape[1] > 1  # one column per cell state
    assert slmm.F.shape == (dd.C.n_obs, 2)  # intercept + crepeat(sex)

    pvs = slmm.interaction_test(dd, exact=True)
    assert pvs.shape == (dd.G.n_vars,)
    assert np.all(np.isfinite(pvs)) and np.all((pvs >= 0) & (pvs <= 1))


def test_structlmm_broadcasts_donor_variants_to_cells():
    """Donor-level variants are expanded to cells exactly as `crepeat()` would."""
    pytest.importorskip("limix_core", reason="StructLMM needs limix-core")
    pytest.importorskip("chiscore", reason="StructLMM needs chiscore")

    from cellink.at.resolver import get_model_matrix
    from cellink.at.structlmm import StructLMM

    dd = _structlmm_donordata(seed=1)
    slmm = StructLMM(y="expr", E="celltype - 1", data=dd, target_level="cell")

    # what the class does internally, against what the resolver's crepeat produces
    broadcast = slmm._variants(dd)
    snp = dd.G.var_names[0]
    dd.G.obs["v0"] = np.asarray(dd.G.X)[:, 0].astype(float)
    by_crepeat = get_model_matrix(dd, "crepeat(v0) - 1", target_level="cell").to_numpy().ravel()

    assert broadcast.shape == (dd.C.n_obs, dd.G.n_vars), f"expected cell rows for {snp}"
    np.testing.assert_allclose(broadcast[:, 0], by_crepeat)


def test_structlmm_donor_level_with_anndata():
    """A plain AnnData needs no target_level, and E can be a donor covariate."""
    pytest.importorskip("limix_core", reason="StructLMM needs limix-core")
    pytest.importorskip("chiscore", reason="StructLMM needs chiscore")
    from cellink.at.structlmm import StructLMM

    rng = np.random.default_rng(2)
    gdata = sim_gdata(n_donors=60, n_snps=3)
    gdata.obs["pheno"] = rng.standard_normal(gdata.n_obs)
    gdata.obs["env"] = rng.standard_normal(gdata.n_obs)

    slmm = StructLMM(y="pheno", E="env", data=gdata)
    pvs = slmm.interaction_test(gdata, exact=True)
    assert pvs.shape == (gdata.n_vars,)
    assert np.all(np.isfinite(pvs))


def test_structlmm_rejects_arrays():
    """Arrays are no longer accepted, and the error says what to pass instead."""
    pytest.importorskip("limix_core", reason="StructLMM needs limix-core")
    from cellink.at.structlmm import StructLMM

    rng = np.random.default_rng(3)
    gdata = sim_gdata(n_donors=30, n_snps=3)
    gdata.obs["pheno"] = rng.standard_normal(gdata.n_obs)
    gdata.obs["env"] = rng.standard_normal(gdata.n_obs)

    with pytest.raises(AssertionError, match="y must be a string"):
        StructLMM(y=rng.standard_normal((30, 1)), E="env", data=gdata)
    with pytest.raises(AssertionError, match="E must be a string"):
        StructLMM(y="pheno", E=rng.standard_normal((30, 2)), data=gdata)

    slmm = StructLMM(y="pheno", E="env", data=gdata)
    with pytest.raises(TypeError, match="Wrap a derived matrix in an AnnData"):
        slmm.interaction_test(rng.standard_normal((30, 3)))
