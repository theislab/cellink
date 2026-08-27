from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellink.tl.external._sclinker import _maybe_remap_ensg_to_hgnc, compute_nmf_programs
from cellink.tl.external._sclinker_utils import (
    SCLINKER_ENHANCER_LINKS_GENOME_BUILD,
    _check_build_consistency,
    _normalize_sclinker_build,
    bedgraph_to_snp_annotation,
    genescores_to_100kb_bedgraph,
)


@pytest.mark.parametrize(
    "raw,expected",
    [("GRCh37", "GRCh37"), ("hg19", "GRCh37"), ("build37", "GRCh37"), ("GRCh38", "GRCh38"), ("hg38", "GRCh38")],
)
def test_normalize_sclinker_build_aliases(raw, expected):
    assert _normalize_sclinker_build(raw) == expected


def test_normalize_sclinker_build_rejects_unknown():
    with pytest.raises(ValueError, match="Invalid genome_build"):
        _normalize_sclinker_build("GRCh99")


def _bim(chrom: str, positions: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CHR": [chrom] * len(positions),
            "SNP": [f"rs{i}" for i in range(len(positions))],
            "CM": [0.0] * len(positions),
            "BP": positions,
            "A1": ["A"] * len(positions),
            "A2": ["G"] * len(positions),
        }
    )


def test_check_build_consistency_raises_on_declared_mismatch():
    bg = pd.DataFrame({"chr": ["21"], "start": [1_000_000], "end": [1_100_000], "score": [1.0]})
    bg.attrs["genome_build"] = "GRCh38"
    bim = _bim("21", [1_000_050, 1_050_000])
    with pytest.raises(ValueError, match="Genome-build mismatch"):
        _check_build_consistency(bg, bim, bim_genome_build="GRCh37")


def test_check_build_consistency_passes_when_declared_builds_match():
    bg = pd.DataFrame({"chr": ["21"], "start": [1_000_000], "end": [1_100_000], "score": [1.0]})
    bg.attrs["genome_build"] = "GRCh37"
    bim = _bim("21", [1_000_050, 1_050_000])
    _check_build_consistency(bg, bim, bim_genome_build="GRCh37")  # should not raise


def test_check_build_consistency_warn_mode_does_not_raise(caplog):
    bg = pd.DataFrame({"chr": ["21"], "start": [1_000_000], "end": [1_100_000], "score": [1.0]})
    bg.attrs["genome_build"] = "GRCh38"
    bim = _bim("21", [1_000_050])
    _check_build_consistency(bg, bim, bim_genome_build="GRCh37", on_mismatch="warn")
    assert any("Genome-build mismatch" in r.message for r in caplog.records)


def test_check_build_consistency_gross_mismatch_fallback_without_declared_build():
    bg = pd.DataFrame({"chr": ["21"], "start": [90_000_000], "end": [90_100_000], "score": [1.0]})
    bim = _bim("21", [1_000_050, 1_050_000, 1_100_000])
    with pytest.raises(ValueError, match="Possible genome-build mismatch"):
        _check_build_consistency(bg, bim)


def test_genescores_to_100kb_bedgraph_propagates_genome_build():
    ga = pd.DataFrame({"GENE": ["SOD1"], "CHR": ["21"], "START": [33031935], "END": [33041244]})
    ga.attrs["genome_build"] = "GRCh37"
    genescores = pd.DataFrame({"program1": [0.9]}, index=["SOD1"])
    bg = genescores_to_100kb_bedgraph(genescores, ga, use_bedtools_for_merge=False)["program1"]
    assert bg.attrs.get("genome_build") == "GRCh37"


def test_sclinker_enhancer_links_build_is_grch38():
    assert SCLINKER_ENHANCER_LINKS_GENOME_BUILD == "GRCh38"


def test_bedgraph_to_snp_annotation_end_to_end_build_mismatch(tmp_path):
    bg = pd.DataFrame({"chr": ["21"], "start": [1_000_000], "end": [1_100_000], "score": [1.0]})
    bg.attrs["genome_build"] = "GRCh38"
    bim = _bim("21", [1_000_050, 1_050_000])
    bim_file = tmp_path / "test.bim"
    bim.to_csv(bim_file, sep="\t", header=False, index=False)

    with pytest.raises(ValueError, match="Genome-build mismatch"):
        bedgraph_to_snp_annotation(bg, bim_file, str(tmp_path / "out"), bim_genome_build="GRCh37")

    out = bedgraph_to_snp_annotation(bg, bim_file, str(tmp_path / "out"), bim_genome_build="GRCh37", on_build_mismatch="ignore")
    assert out.exists()


def test_maybe_remap_ensg_to_hgnc_remaps_when_gene_name_present():
    H = pd.DataFrame({"NMF_0": [1.0, 2.0]}, index=["ENSG00000142168", "ENSG00000142192"])
    var = pd.DataFrame({"gene_name": ["SOD1", "APP"]}, index=["ENSG00000142168", "ENSG00000142192"])
    adata = AnnData(np.zeros((2, 2)), var=var)
    remapped, did_remap = _maybe_remap_ensg_to_hgnc({"H": H}, adata)
    assert did_remap
    assert set(remapped["H"].index) == {"SOD1", "APP"}


def test_maybe_remap_ensg_to_hgnc_warns_without_gene_name_column(caplog):
    H = pd.DataFrame({"NMF_0": [1.0, 2.0]}, index=["ENSG00000142168", "ENSG00000142192"])
    adata = AnnData(np.zeros((2, 2)), var=pd.DataFrame(index=["ENSG00000142168", "ENSG00000142192"]))
    remapped, did_remap = _maybe_remap_ensg_to_hgnc({"H": H}, adata)
    assert not did_remap
    assert list(remapped["H"].index) == ["ENSG00000142168", "ENSG00000142192"]
    assert any("no 'gene_name' column" in r.message for r in caplog.records)


def test_maybe_remap_ensg_to_hgnc_noop_when_already_symbols():
    H = pd.DataFrame({"NMF_0": [1.0, 2.0]}, index=["SOD1", "APP"])
    adata = AnnData(np.zeros((2, 2)), var=pd.DataFrame(index=["SOD1", "APP"]))
    remapped, did_remap = _maybe_remap_ensg_to_hgnc({"H": H}, adata)
    assert not did_remap
    assert list(remapped["H"].index) == ["SOD1", "APP"]


def test_compute_nmf_programs_remaps_ensg_var_names():
    rng = np.random.default_rng(0)
    ensg = ["ENSG00000142192", "ENSG00000142168", "ENSG00000159216"]
    symbols = {"ENSG00000142192": "APP", "ENSG00000142168": "SOD1", "ENSG00000159216": "RUNX1"}
    n_cells = 60
    X = rng.poisson(2.0, size=(n_cells, len(ensg))).astype(np.float32)
    var = pd.DataFrame({"gene_name": [symbols[g] for g in ensg]}, index=ensg)
    obs = pd.DataFrame({"cell_type": rng.choice(["A", "B"], size=n_cells)}, index=[f"c{i}" for i in range(n_cells)])
    adata = AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X

    _, H, _ = compute_nmf_programs(adata, n_components=2, celltype_col="cell_type", layer="counts", save=False)

    assert set(H.index) == set(symbols.values())
    assert not any(g.startswith("ENSG") for g in H.index)
