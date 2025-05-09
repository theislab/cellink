from pathlib import Path

import numpy as np
import pandas_plink as pdp

from cellink.at import utils

DATA = Path("tests/data")


def test_generate_phenotype_data():
    """Test the generate_phenotype_data function."""
    # Generate phenotype data
    N = 10000  # number of individuals
    S = 100  # number of variants
    number_causal_variants = 10  # number of causal variants
    vg = 0.5  # variance explained by the causal variants

    X = np.random.choice([0, 1, 2], size=(N, S))
    X = np.asarray(X, dtype=np.float64)

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    Y, betas, Yg, Yn = utils.generate_phenotype(X, vg=vg, number_causal_variants=number_causal_variants)

    assert Y.shape == (N, 1), "Generated phenotype data shape is incorrect"

    # Test the mean and standard deviation of the generated phenotype data
    np.testing.assert_allclose(
        Y.mean(axis=0),
        0,
        rtol=1e-2,
        atol=1e-2,
        err_msg="Generated phenotype data mean is incorrect",
    )

    np.testing.assert_allclose(
        Y.std(axis=0),
        1,
        rtol=1e-2,
        atol=1e-2,
        err_msg="Generated phenotype data std is incorrect",
    )

    assert betas.shape == (S, 1), "Generated beta data shape is incorrect"

    np.testing.assert_allclose(
        Yg.var(0),
        vg,
        rtol=1e-2,
        atol=1e-2,
        err_msg="Generated Yg data variance is incorrect",
    )
    np.testing.assert_allclose(
        Yg.mean(0),
        0,
        rtol=1e-2,
        atol=1e-2,
        err_msg="Generated Yg data mean is incorrect",
    )

    np.testing.assert_allclose(
        Yn.var(0),
        1 - vg,
        rtol=1e-2,
        atol=1e-2,
        err_msg="Generated Yn data variance is incorrect",
    )
    np.testing.assert_allclose(
        Yn.mean(0),
        0,
        rtol=1e-2,
        atol=1e-2,
        err_msg="Generated Yn data mean is incorrect",
    )


def test_burden_testing():
    _, _, bed = pdp.read_plink(DATA / "simulated_genotype_calls", verbose=False)
    X = bed.compute().T
    X = utils.ensure_float64_array(X)
    X = utils.xgower_factor_(X)

    Y, betas = utils.generate_phenotype(X, vg=0.3, number_causal_variants=10)

    gwas = utils.GWAS(Y)
    g = np.sum(X, axis=1)  # define the burden as the sum of the variants (arbitrary)
    g = utils.ensure_float64_array(g)
    pv = gwas.process(g)
    assert pv is not None, "P-value is None"
    Y = np.permutation(Y)
    gwas = utils.GWAS(Y)
    gwas.process(g)
    pvp = gwas.process(g)  # pv permutated
    assert pv < pvp, "P-value is not smaller than permutated p-value"


if __name__ == "__main__":
    # test_generate_phenotype_data()
    test_burden_testing()
