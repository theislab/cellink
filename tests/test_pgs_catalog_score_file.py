from __future__ import annotations

import gzip

import pytest

from cellink.resources import get_pgs_catalog_score_file

SUBMITTED_URL = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000001/ScoringFiles/PGS000001.txt.gz"
HARMONIZED_URLS = {
    "GRCh37": {"positions": "https://example.org/PGS000001_hmPOS_GRCh37.txt.gz"},
    "GRCh38": {"positions": "https://example.org/PGS000001_hmPOS_GRCh38.txt.gz"},
}
SCORING_FILE = (
    "###PGS CATALOG SCORING FILE\n"
    "#pgs_id=PGS000001\n"
    "#HmPOS_build=GRCh38\n"
    "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\thm_chr\thm_pos\n"
    "rs1\t1\t100\tA\tG\t0.1\t1\t1100\n"
    "rs2\t22\t200\tC\tT\t-0.2\t22\t1200\n"
)


@pytest.fixture
def pgs_catalog(monkeypatch):
    """Stand in for the PGS Catalog REST API and FTP, recording every download."""
    calls = {"fetch": 0, "downloads": []}

    def fake_fetch(url, params=None, paginate=False):
        calls["fetch"] += 1
        return {"id": "PGS000001", "ftp_scoring_file": SUBMITTED_URL, "ftp_harmonized_scoring_files": HARMONIZED_URLS}

    def fake_urlretrieve(url, dest):
        calls["downloads"].append(url)
        with gzip.open(dest, "wt") as fh:
            fh.write(SCORING_FILE)

    monkeypatch.setattr("cellink.resources._gwas_prs_qtl._fetch", fake_fetch)
    monkeypatch.setattr("cellink.resources._gwas_prs_qtl.urlretrieve", fake_urlretrieve)
    return calls


def test_genome_build_downloads_the_harmonized_file(tmp_path, pgs_catalog):
    df = get_pgs_catalog_score_file("PGS000001", genome_build="hg38", data_home=tmp_path)

    assert pgs_catalog["downloads"] == [HARMONIZED_URLS["GRCh38"]["positions"]]
    assert (tmp_path / "PGS000001_hmPOS_GRCh38_scoring_file.txt.gz").exists()
    assert list(df["hm_pos"]) == [1100, 1200]


def test_without_genome_build_downloads_the_submitted_file(tmp_path, pgs_catalog):
    path = get_pgs_catalog_score_file("PGS000001", data_home=tmp_path, return_path=True)

    assert pgs_catalog["downloads"] == [SUBMITTED_URL]
    assert path == tmp_path / "PGS000001_scoring_file.txt.gz"


def test_existing_file_is_reused_unless_refresh(tmp_path, pgs_catalog):
    first = get_pgs_catalog_score_file("PGS000001", genome_build="GRCh37", data_home=tmp_path, return_path=True)
    second = get_pgs_catalog_score_file("PGS000001", genome_build="GRCh37", data_home=tmp_path, return_path=True)

    assert first == second
    assert pgs_catalog["downloads"] == [HARMONIZED_URLS["GRCh37"]["positions"]]
    assert pgs_catalog["fetch"] == 1

    get_pgs_catalog_score_file("PGS000001", genome_build="GRCh37", data_home=tmp_path, refresh=True)
    assert len(pgs_catalog["downloads"]) == 2


def test_missing_harmonized_build_raises(tmp_path, pgs_catalog, monkeypatch):
    monkeypatch.delitem(HARMONIZED_URLS, "GRCh38")

    with pytest.raises(ValueError, match="No harmonized GRCh38 scoring file"):
        get_pgs_catalog_score_file("PGS000001", genome_build="GRCh38", data_home=tmp_path)
    assert pgs_catalog["downloads"] == []
