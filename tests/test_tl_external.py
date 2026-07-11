import numpy as np
import pandas as pd

from cellink.tl.external import JointNMFWrapper, compute_escore, scores_to_covar, scores_to_gmt


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
