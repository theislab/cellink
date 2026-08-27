from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from cellink.tl.external._ldsc import munge_sumstats
from cellink.tl.external._sldsc_utils import preprocess_for_sldsc


def test_munge_sumstats_snplist_raises_clear_error():
    with pytest.raises(ValueError, match="snplist"):
        munge_sumstats(sumstats_file="unused.tsv", snplist="some_list.txt", run=False)


def test_munge_sumstats_creates_missing_output_directory(tmp_path, monkeypatch):
    sumstats = tmp_path / "in.tsv"
    sumstats.write_text("SNP\tA1\tA2\tP\tBETA\nrs1\tA\tG\t0.01\t0.1\n")

    class DummyRunner:
        munge_command = "munge_sumstats.py"

        def run_command(self, cmd, file_paths=None, check=True):
            self.cmd = cmd

    out_prefix = str(tmp_path / "nested" / "dir" / "out")
    assert not (tmp_path / "nested" / "dir").exists()
    munge_sumstats(sumstats_file=str(sumstats), out_prefix=out_prefix, runner=DummyRunner())
    assert (tmp_path / "nested" / "dir").exists()


def _mhc_test_adata(chrom_labels):
    var = pd.DataFrame(
        {
            "gene": ["HLA-DRB1", "SOD1"],
            "chrom": chrom_labels,
            "start": [32546547, 33031935],
            "end": [32557613, 33041244],
            "gene_biotype": ["protein_coding", "protein_coding"],
        },
        index=["HLA-DRB1", "SOD1"],
    )
    rng = np.random.default_rng(0)
    n_cells = 40
    X = rng.poisson(3.0, size=(n_cells, 2)).astype(np.float32)
    obs = pd.DataFrame({"cell_type": rng.choice(["A", "B"], size=n_cells)}, index=[f"c{i}" for i in range(n_cells)])
    return AnnData(X=X, obs=obs, var=var)


@pytest.mark.parametrize("chroms", [["6", "21"], ["chr6", "chr21"], ["CHR6", "Chr21"]])
def test_preprocess_for_sldsc_mhc_filter_excludes_real_mhc_gene(chroms):
    adata = _mhc_test_adata(chroms)
    adata_filt, _, _ = preprocess_for_sldsc(
        adata,
        celltype_col="cell_type",
        log_transform=True,
        filter_protein_coding=True,
        filter_expressed=False,
        filter_mhc=True,
        fetch_annotation=False,
        inplace=False,
    )
    genes_left = list(adata_filt.var_names)
    assert "HLA-DRB1" not in genes_left, "MHC gene should be excluded when filter_mhc=True"
    assert "SOD1" in genes_left


def test_preprocess_for_sldsc_mhc_filter_disabled_keeps_mhc_gene():
    adata = _mhc_test_adata(["6", "21"])
    adata_filt, _, _ = preprocess_for_sldsc(
        adata,
        celltype_col="cell_type",
        log_transform=True,
        filter_protein_coding=True,
        filter_expressed=False,
        filter_mhc=False,
        fetch_annotation=False,
        inplace=False,
    )
    assert "HLA-DRB1" in list(adata_filt.var_names)
