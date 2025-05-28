import logging
from pathlib import Path

import numpy as np

from cellink.at import utils
from cellink.at.acat import compute_acat
from cellink.at.gwas import GWAS
from cellink.at.skat import Skat
from cellink.at.structlmm import StructLMM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA = Path("/Users/antonio.nappi/Desktop/sc-genetics/tests/data")


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
    N = 100000  # number of individuals
    S = 300  # number of variants
    number_causal_variants = 50  # number of causal variants
    vg = 0.3  # variance explained by the causal variants

    X = np.random.choice([0, 1, 2], size=(N, S), p=[0.99, 0.005, 0.005])
    X = np.asarray(X, dtype=np.float64)

    Y, *_ = utils.generate_phenotype(X, vg=vg, number_causal_variants=number_causal_variants)

    gwas = GWAS(Y)
    g = np.sum(X, axis=1, keepdims=True)  # define the burden as the sum of the variants (arbitrary)

    g = utils.ensure_float64_array(g)
    gwas.test_association(g)
    pv = gwas.getPv()
    assert pv is not None, "P-value is None"
    Y = np.random.permutation(Y)
    gwas = GWAS(Y)
    gwas.test_association(g)
    pvp = gwas.getPv()  # pv permutated

    assert pv < pvp, "P-value is not smaller than permutated p-value"


def test_skat_testing():
    """Test the burden testing function."""
    N = 10000  # number of individuals
    S = 10  # number of variants
    number_causal_variants = 3  # number of causal variants
    vg = 0.1  # variance explained by the causal variants

    X = np.random.choice([0, 1, 2], size=(N, S), p=[0.9, 0.05, 0.05])
    X = np.asarray(X, dtype=np.float64)
    logger.info(f"number of variants: {X.sum(1)}")
    Y, *_ = utils.generate_phenotype(X, vg=vg, number_causal_variants=number_causal_variants)

    skat = Skat(min_threshold=10)
    pv = skat.run_test(Y=Y, X=X)
    assert pv is not None, "P-value is None"

    Y = np.random.permutation(Y)
    pvp = skat.run_test(Y=Y, X=X)  # pv permutated
    logger.info(f"pv: {pv}, pvp: {pvp}")

    assert pv < pvp, "P-value is not smaller than permutated p-value"


def test_acat_testing():
    N = 10000  # number of individuals
    S = 300  # number of variants
    number_causal_variants = 50  # number of causal variants
    vg = 0.15  # variance explained by the causal variants

    X = np.random.choice([0, 1, 2], size=(N, S), p=[0.9, 0.05, 0.05])
    X = np.asarray(X, dtype=np.float64)
    Y, *_ = utils.generate_phenotype(X, vg=vg, number_causal_variants=number_causal_variants)

    skat = Skat(min_threshold=10)

    skat_pv = skat.run_test(Y=Y, X=X)

    Yp = np.random.permutation(Y)
    skat_pvp = skat.run_test(Y=Yp, X=X)  # pv permutated
    X_std = (X - X.mean(0)) / utils.xgower_factor_(X)
    g = np.sum(X_std, axis=1, keepdims=True)  # define the burden as the sum of the variants (arbitrary)

    g = utils.ensure_float64_array(g)
    gwas = GWAS(Y)
    gwas.test_association(g)
    burden_pv = gwas.getPv()

    gwas = GWAS(Yp)
    gwas.test_association(g)
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
    print(f"acat_pv: {acat_pv}, acat_pvp: {acat_pvp}")
    assert acat_pv < acat_pvp, "P-value is not smaller than permutated p-value"


def test_structlmm():
    N = 3000
    S = 1
    Edim = 10
    vg = 0.5
    number_causal_variants = 1
    number_causal_interactions = 5
    percentage_interaction = 0.5

    # Simulate data
    X = np.random.choice([0, 1, 2], size=(N, S)).astype(np.float64)
    E = np.random.randn(N, Edim)

    Y, betas, Yg, Yn, Ye, gammas = utils.generate_phenotype(
        X,
        vg=vg,
        number_causal_variants=number_causal_variants,
        E=E,
        number_causal_interactions=number_causal_interactions,
        percentage_variance_explained_by_interaction=percentage_interaction,
    )

    F = np.ones((N, 1))  # intercept

    # --- Full interaction test (StructLMM)
    model_int = StructLMM(Y=Y, X=X, E=E, F=F)
    pvals_interaction = model_int.interaction_test()

    # --- Simple additive test (StructLMM with E = 1)
    dummy_E = np.ones((N, 1))
    model_add = StructLMM(Y=Y, X=X, E=dummy_E, F=F)
    pvals_additive = model_add.interaction_test()

    # --- Compare p-values
    true_causal_idx = np.where(betas.ravel() != 0)[0]

    power_add = np.mean(pvals_additive[true_causal_idx] < 0.05)
    power_int = np.mean(pvals_interaction[true_causal_idx] < 0.05)

    print(f"🔍 Power (StructLMM additive):    {power_add:.2%}")
    print(f"🔍 Power (StructLMM interaction): {power_int:.2%}")

    # --- Permutation test for Type I error (interaction only)
    Y_perm = np.random.permutation(Y)
    model_perm = StructLMM(Y=Y_perm, X=X, E=E, F=F)
    perm_pvals = model_perm.interaction_test()

    print(f"⚖️  Lambda inflation under perm: {np.median(perm_pvals) / 0.5:.2f}")


if __name__ == "__main__":
    logger.info("Running tests...")
    logger.info("Testing generate_phenotype_data...")
    test_generate_phenotype_data()
    logger.info("Testing burden testing...")
    test_burden_testing()
    logger.info("Testing skat testing...")
    test_skat_testing()
    logger.info("Testing acat testing...")
    test_acat_testing()
    logger.info("Testing structlmm testing...")
    test_structlmm()
