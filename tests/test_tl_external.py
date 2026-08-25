import numpy as np
import pandas as pd
import pytest

from cellink.tl.external import JointNMFWrapper, compute_escore, scores_to_covar, scores_to_gmt
from cellink.tl.external._tensorqtl import build_known_cis_eqtls_from_tensorqtl


def test_scores_to_gmt(tmp_path):
    scores = pd.DataFrame(
        {
            "CD4_T": [5.0, 4.0, 3.0, 2.0, 1.0],
            "CD8_T": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"],
    )

    out_file = scores_to_gmt(scores, tmp_path / "sets.gmt", top_frac=0.4)
    lines = out_file.read_text().strip().split("\n")

    assert len(lines) == 2
    sets = {line.split("\t")[0]: line.split("\t")[2:] for line in lines}
    assert set(sets["CD4_T"]) == {"ENSG1", "ENSG2"}
    assert set(sets["CD8_T"]) == {"ENSG4", "ENSG5"}


def test_scores_to_gmt_ascending(tmp_path):
    scores = pd.DataFrame({"CD4_T": [5.0, 4.0, 3.0, 2.0, 1.0]}, index=["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"])

    out_file = scores_to_gmt(scores, tmp_path / "bottom.gmt", top_frac=0.4, ascending=True)
    line = out_file.read_text().strip()
    genes = line.split("\t")[2:]
    assert set(genes) == {"ENSG4", "ENSG5"}


def test_scores_to_covar(tmp_path):
    scores = pd.DataFrame(
        {"CD4 T": [0.5, -0.2, np.nan], "CD8-T": [0.1, 0.3, 0.4]},
        index=["ENSG1", "ENSG2", "ENSG3"],
    )

    out_file = scores_to_covar(scores, tmp_path / "out.covar")
    covar = pd.read_csv(out_file, sep="\t", index_col="GENE", na_values="NA")

    assert list(covar.index) == ["ENSG1", "ENSG2", "ENSG3"]
    assert "CD4_T" in covar.columns and "CD8-T" in covar.columns
    assert pd.isna(covar.loc["ENSG3", "CD4_T"])
    np.testing.assert_allclose(covar.loc["ENSG1", "CD8-T"], 0.1)


def test_scores_to_covar_negate(tmp_path):
    scores = pd.DataFrame({"CD4_T": [0.5, -0.2]}, index=["ENSG1", "ENSG2"])

    out_file = scores_to_covar(scores, tmp_path / "out.covar", negate=True)
    covar = pd.read_csv(out_file, sep="\t", index_col="GENE")

    np.testing.assert_allclose(covar.loc["ENSG1", "CD4_T"], -0.5)
    np.testing.assert_allclose(covar.loc["ENSG2", "CD4_T"], 0.2)


def test_compute_escore():
    results = pd.DataFrame(
        {
            "trait": ["height", "height", "bmi", "bmi"],
            "strategy": ["s1", "s1", "s1", "s1"],
            "program": ["ProgramA", "AllCoding", "ProgramA", "AllCoding"],
            "Enrichment": [2.0, 1.0, 3.0, 1.5],
            "Enrichment_std_error": [0.1, 0.1, 0.2, 0.2],
        }
    )

    scored = compute_escore(results)
    program_a = scored[scored["program"] == "ProgramA"].set_index("trait")

    np.testing.assert_allclose(program_a.loc["height", "E_score"], 1.0)
    np.testing.assert_allclose(program_a.loc["bmi", "E_score"], 1.5)
    np.testing.assert_allclose(program_a.loc["height", "E_score_se"], np.sqrt(0.1**2 + 0.1**2))


def test_joint_nmf_wrapper():
    rng = np.random.default_rng(0)
    Xh = rng.random((30, 12))
    Xd = rng.random((30, 12))

    wrapper = JointNMFWrapper(
        Xh,
        Xd,
        n_shared=2,
        n_healthy_specific=1,
        n_disease_specific=1,
        n_init=1,
        max_iters=20,
        random_state=0,
    ).fit()

    assert wrapper.Wh.shape == (30, 3)
    assert wrapper.Wd.shape == (30, 3)
    assert wrapper.Hh.shape == (3, 12)
    assert wrapper.Hd.shape == (3, 12)
    assert (wrapper.Wh >= 0).all()
    assert (wrapper.Hh >= 0).all()


def _write_tensorqtl_nominal_parquet(path, rows):
    pd.DataFrame(rows, columns=["gene", "variant_id", "pval"]).to_parquet(path)
    return path


def test_build_known_cis_eqtls_picks_lowest_pval_per_gene(tmp_path):
    parquet = _write_tensorqtl_nominal_parquet(
        tmp_path / "nominal.parquet",
        [
            ("ENSG1", "1:100:A:G", 0.5),
            ("ENSG1", "1:200:A:G", 0.01),  # lowest p for ENSG1
            ("ENSG2", "2:100:A:G", 0.2),
            ("ENSG2", "2:200:A:G", 0.9),
        ],
    )
    known = build_known_cis_eqtls_from_tensorqtl(str(parquet), gene_names=["ENSG1", "ENSG2"])

    # only the winning (lowest-pval) SNP per gene becomes a row at all: this is a
    # sparse "selected loci" matrix, not a full candidate universe with explicit 0s
    assert known.shape == (2, 2)
    assert known.loc["1:200:A:G", "ENSG1"] == 1
    assert "1:100:A:G" not in known.index  # the higher-pval candidate never appears
    assert known.loc["2:100:A:G", "ENSG2"] == 1
    assert known.values.sum() == 2  # exactly one selected SNP per gene


def test_build_known_cis_eqtls_respects_pval_threshold(tmp_path):
    parquet = _write_tensorqtl_nominal_parquet(
        tmp_path / "nominal.parquet",
        [
            ("ENSG1", "1:100:A:G", 0.5),
            ("ENSG2", "2:100:A:G", 0.2),
        ],
    )
    # ENSG1's only variant fails the threshold and must not appear at all
    known = build_known_cis_eqtls_from_tensorqtl(
        str(parquet), gene_names=["ENSG1", "ENSG2"], pval_threshold=0.3,
    )
    assert "ENSG1" not in known.columns
    assert known.loc["2:100:A:G", "ENSG2"] == 1


def test_build_known_cis_eqtls_max_snps_per_gene(tmp_path):
    parquet = _write_tensorqtl_nominal_parquet(
        tmp_path / "nominal.parquet",
        [
            ("ENSG1", "1:100:A:G", 0.5),
            ("ENSG1", "1:200:A:G", 0.01),
            ("ENSG1", "1:300:A:G", 0.02),
        ],
    )
    known = build_known_cis_eqtls_from_tensorqtl(str(parquet), gene_names=["ENSG1"], max_snps_per_gene=2)
    assert known["ENSG1"].sum() == 2
    assert "1:100:A:G" not in known.index  # the worst of the 3 is still excluded
    assert set(known.index) == {"1:200:A:G", "1:300:A:G"}


def test_build_known_cis_eqtls_raises_when_nothing_survives(tmp_path):
    parquet = _write_tensorqtl_nominal_parquet(
        tmp_path / "nominal.parquet",
        [("ENSG1", "1:100:A:G", 0.9)],
    )
    with pytest.raises(ValueError, match="No cis-eQTL pairs survived"):
        build_known_cis_eqtls_from_tensorqtl(str(parquet), gene_names=["ENSG1"], pval_threshold=0.01)
