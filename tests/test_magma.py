from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellink.tl.external import _magma
from cellink.tl.external._magma import prepare_magma_inputs

TINY = 2.2250738585072014e-308


@pytest.fixture(autouse=True)
def _no_reference_download(monkeypatch, tmp_path):
    gene_loc = tmp_path / "genes.loc"
    gene_loc.write_text("")
    monkeypatch.setattr(_magma, "download_magma_references", lambda **kwargs: (gene_loc, None))


def _gwas():
    # harmonised GWAS Catalog layout: no SNP IDs, effect allele = ALT for the first three SNPs
    return pd.DataFrame(
        {
            "chromosome": [1, 1, 1, 2, 2],
            "base_pair_location": [100, 200, 300, 400, 400],
            "effect_allele": ["g", "T", "C", "A", "G"],
            "other_allele": ["a", "C", "G", "G", "A"],
            "p_value": [0.5, 2.64e-319, 0.0, 1e-5, 1e-3],
            "variant_id": [np.nan] * 5,
        }
    )


_MAPPING = {"chromosome": "CHR", "base_pair_location": "BP", "p_value": "P"}


def _write_bim(prefix, ids):
    pd.DataFrame({"c": 1, "id": ids, "cm": 0, "bp": 1, "a1": "A", "a2": "G"}).to_csv(
        f"{prefix}.bim", sep="\t", header=False, index=False
    )
    for ext in (".bed", ".fam"):
        open(f"{prefix}{ext}", "w").close()


def _read(path):
    return pd.read_csv(path, sep="\t", header=None, dtype={0: str})


def test_subnormal_and_zero_p_values_are_clipped(tmp_path):
    _, pval_file, _, _ = prepare_magma_inputs(_gwas(), output_prefix=str(tmp_path / "t"), col_mapping=_MAPPING)
    p = _read(pval_file)[1].to_numpy()
    assert p.tolist() == [0.5, TINY, TINY, 1e-5, 1e-3]
    assert p.min() >= np.finfo(np.float64).tiny
    assert "e-319" not in pval_file.read_text()


def test_swapped_allele_order_matches_external_bim(tmp_path):
    # .bim IDs are chr_pos_ref_alt; the GWAS ID 2_400_A_G exists as is, and its swap 2_400_G_A is also
    # a GWAS SNP, so neither of the chr 2 rows may be renamed
    ref = tmp_path / "1000G.EUR.QC"
    _write_bim(ref, ["1_100_A_G", "1_200_C_T", "1_999_A_C", "2_400_A_G"])
    snp_loc, pval_file, _, ld_ref = prepare_magma_inputs(
        _gwas(), output_prefix=str(tmp_path / "t"), col_mapping=_MAPPING, ld_source="external", external_ld_prefix=ref
    )
    assert str(ld_ref) == str(ref)  # ".QC" stays part of the prefix
    ids = _read(pval_file)[0].tolist()
    assert ids == ["1_100_A_G", "1_200_C_T", "1_300_C_G", "2_400_A_G", "2_400_G_A"]
    assert _read(snp_loc)[0].tolist() == ids


def test_rsid_input_is_left_alone(tmp_path):
    ref = tmp_path / "ref"
    _write_bim(ref, ["rs1"])
    gwas = _gwas().assign(variant_id=["rs1", "rs2", "rs3", "rs4", "rs5"])
    _, pval_file, _, _ = prepare_magma_inputs(
        gwas,
        output_prefix=str(tmp_path / "t"),
        col_mapping={**_MAPPING, "variant_id": "SNP"},
        ld_source="external",
        external_ld_prefix=ref,
    )
    assert _read(pval_file)[0].tolist() == ["rs1", "rs2", "rs3", "rs4", "rs5"]


def test_donor_data_ld_reference_matches_alt_effect_alleles(tmp_path):
    ids = ["1_100_A_G", "1_200_C_T", "1_300_G_C"]
    var = pd.DataFrame(
        {"chrom": ["1"] * 3, "pos": [100, 200, 300], "a0": ["A", "C", "G"], "a1": ["G", "T", "C"]}, index=ids
    )
    G = AnnData(
        X=np.array([[0, 1, 2], [1, 1, 0], [2, 0, 1], [1, 2, 1]], dtype=np.float32),
        obs=pd.DataFrame(index=[f"d{i}" for i in range(4)]),
        var=var,
    )
    _, pval_file, _, ld_ref = prepare_magma_inputs(
        _gwas(),
        output_prefix=str(tmp_path / "t"),
        col_mapping=_MAPPING,
        ld_source="dd_genotypes",
        dd=SimpleNamespace(G=G),
    )
    bim_ids = set(pd.read_csv(f"{ld_ref}.bim", sep=r"\s+", header=None, dtype=str)[1])
    gwas_ids = _read(pval_file)[0].tolist()
    assert set(gwas_ids[:3]) <= bim_ids
