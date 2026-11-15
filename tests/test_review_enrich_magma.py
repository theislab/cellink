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


_FAKE_MAGMA = r'''#!/usr/bin/env python3
"""Stand-in for MAGMA: --batch i N writes <out>.batch<i>_<N>.genes.out, --merge concatenates them.
Like MAGMA, it allows at most MAX_BATCHES batches and reports the surplus ones as empty."""
import glob, os, sys
MAX_BATCHES = int(os.environ.get("FAKE_MAGMA_MAX_BATCHES", "8"))
args = sys.argv[1:]
out = args[args.index("--out") + 1]
if "--merge" in args:
    prefix = args[args.index("--merge") + 1]
    files = sorted(glob.glob(prefix + ".batch*_*.genes.out"))
    if len(files) < 2:
        sys.exit("ERROR - merging gene results files: only a single file specified")
    rows = [line for f in files for line in open(f).read().splitlines()[1:]]
    open(out + ".genes.out", "w").write("GENE P\n" + "\n".join(rows) + "\n")
    open(out + ".genes.raw", "w").write("raw\n")
    sys.exit(0)
i, n = (int(x) for x in args[args.index("--batch") + 1 : args.index("--batch") + 3])
if n > MAX_BATCHES:
    n = MAX_BATCHES
    if i > n:
        print(f"ERROR - configuration: maximum number of batches for input data is {n}, batch {i} is empty")
        sys.exit(1)
open(f"{out}.batch{i}_{n}.genes.out", "w").write(f"GENE P\ng{i} 0.{i}\n")
open(f"{out}.batch{i}_{n}.genes.raw", "w").write("raw\n")
'''


@pytest.mark.parametrize(("max_batches", "n_genes"), [(8, 3), (2, 2), (1, 1)])
def test_magma_gene_analysis_runs_batches_and_merges(tmp_path, monkeypatch, max_batches, n_genes):
    from cellink.tl.external._magma import run_magma_gene_analysis

    magma = tmp_path / "magma"
    magma.write_text(_FAKE_MAGMA)
    magma.chmod(0o755)
    monkeypatch.setenv("FAKE_MAGMA_MAX_BATCHES", str(max_batches))
    out = run_magma_gene_analysis(
        tmp_path / "p.txt", tmp_path / "a.genes.annot", tmp_path / "ref", str(tmp_path / "res"),
        n_samples=100, magma_bin=str(magma), n_jobs=3,
    )  # fmt: skip
    genes = pd.read_csv(out, sep=" ")
    assert sorted(genes["GENE"]) == [f"g{i}" for i in range(1, n_genes + 1)]
    assert not list(tmp_path.glob("res.batch*"))  # batch files are cleaned up
