from __future__ import annotations

import inspect
import io
import os
import subprocess
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cellink.tl import _annotate_snps_ensembl_rest as rest
from cellink.tl import _annotate_snps_genotype_data as genotype_annotation
from cellink.tl._annotate_snps_genotype_data import run_favor, run_snpeff


class _FakeResponse:
    ok = True

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _fake_get(payload_by_region):
    def get(url, headers=None):
        region = url.split("/overlap/region/human/")[1].split("?")[0]
        return _FakeResponse(payload_by_region.get(region, []))

    return get


# get_snp_df


def test_split_snp_id_accepts_cellink_and_colon_ids():
    assert rest._split_snp_id("1:5:T:C") == ("1", "5", "T", "C")
    assert rest._split_snp_id("1_5_T_C") == ("1", "5", "T", "C")
    assert rest._split_snp_id("chr1_KI270706v1_random_100_A_G") == ("chr1_KI270706v1_random", "100", "A", "G")
    with pytest.raises(ValueError, match="neither"):
        rest._split_snp_id("rs123")


def test_get_snp_df_with_underscore_ids(monkeypatch):
    payload = {
        "1:100-100": [
            {"feature_type": "gene", "biotype": "protein_coding", "id": "ENSG1", "start": 50, "end": 150},
            {"feature_type": "variation", "source": "dbSNP", "id": "rs1", "alleles": ["A", "G"]},
        ]
    }
    monkeypatch.setattr(rest.requests, "get", _fake_get(payload))
    var_df, gene_df = rest.get_snp_df(["1_100_A_G"])
    assert len(var_df) == 1
    assert var_df["rs_id"].tolist() == ["rs1"]
    assert gene_df.index.tolist() == ["ENSG1"]


def test_get_snp_df_without_overlap_returns_empty_frames(monkeypatch):
    monkeypatch.setattr(rest.requests, "get", _fake_get({}))
    var_df, gene_df = rest.get_snp_df(["1_100_A_G", "2:5:C:T"])
    assert var_df.empty
    assert gene_df.empty
    assert gene_df.index.name == "id"


# snpEff / FAVOR


def test_run_snpeff_defaults_to_a_genome_snpeff_ships():
    assert inspect.signature(run_snpeff).parameters["genome_assembly"].default == "GRCh38.99"


def test_run_favor_expands_home_and_matches_string_chromosomes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "favor").mkdir(parents=True)
    pd.DataFrame(
        {
            "variant_vcf": ["1-100-A-G", "1-200-C-T"],
            "chromosome": [1, 1],
            "position": [100, 200],
            "ref_vcf": ["A", "C"],
            "alt_vcf": ["G", "T"],
            "cadd": [1.5, 2.5],
        }
    ).to_csv(home / "favor" / "chr1_part.csv", index=False)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(workdir)
    # the database is in place, so nothing may be downloaded (and tests never touch the network)
    monkeypatch.setattr(genotype_annotation, "download_favor", lambda **kw: pytest.fail("download attempted"))

    var = pd.DataFrame({"chrom": ["1", "1"], "pos": [100, 200], "a0": ["A", "C"], "a1": ["G", "T"]})
    gdata = ad.AnnData(X=np.zeros((2, 2)), var=var.set_index(pd.Index(["1_100_A_G", "1_200_C_T"])))
    annos = run_favor(database_dir="~/favor", G=gdata, return_annos=True)

    assert not os.path.exists(workdir / "~")  # no literal '~' directory
    assert annos["snp_id"].to_list() == ["1_100_A_G", "1_200_C_T"]
    assert annos["cadd"].to_list() == [1.5, 2.5]


def _fake_favor_download(monkeypatch, archive_bytes):
    """Let `wget -O <name> URL` write ``archive_bytes``; every other command runs for real."""
    real_run = subprocess.run

    def run(cmd, *args, cwd=None, **kwargs):
        if cmd[0] == "wget":
            (Path(cwd) / cmd[cmd.index("-O") + 1]).write_bytes(archive_bytes)
            return subprocess.CompletedProcess(cmd, 0)
        return real_run(cmd, *args, cwd=cwd, **kwargs)

    monkeypatch.setattr(genotype_annotation.subprocess, "run", run)


def _favor_archive(tmp_path):
    src = tmp_path / "src"
    (src / "FAVORDB").mkdir(parents=True)
    (src / "FAVORDB" / "chr13_1.csv").write_text("variant_vcf,cadd\n13-1-A-G,1.0\n" * 2000)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        tar.add(src / "FAVORDB", arcname="FAVORDB")
    return buffer.getvalue()


def test_download_favor_moves_complete_extraction_into_place(tmp_path, monkeypatch):
    database = tmp_path / "db"
    database.mkdir()
    (database / "6170503").write_bytes(b"truncated archive from an interrupted run")
    _fake_favor_download(monkeypatch, _favor_archive(tmp_path))

    genotype_annotation.download_favor(chromosome=13, database_dir=str(database))

    assert (database / "FAVORDB" / "chr13_1.csv").read_text().startswith("variant_vcf,cadd")
    # neither the downloaded archive nor the private directory is left behind
    assert sorted(p.name for p in database.iterdir()) == ["6170503", "FAVORDB"]


def test_download_favor_leaves_no_partial_csvs_when_extraction_fails(tmp_path, monkeypatch):
    database = tmp_path / "db"
    database.mkdir()
    _fake_favor_download(monkeypatch, _favor_archive(tmp_path)[:12_000])  # cut mid-file

    with pytest.raises(subprocess.CalledProcessError):
        genotype_annotation.download_favor(chromosome=13, database_dir=str(database))

    assert list(database.iterdir()) == []
