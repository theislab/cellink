from __future__ import annotations

import contextlib

import pandas as pd
import pytest

from cellink.tl.external import _biomart
from cellink.tl.external._biomart import BiomartQueryError, query_biomart

ERROR_PAGE = "<html><body><h1>Service unavailable</h1></body></html>\n"


class _Response:
    def __init__(self, text, url):
        self.text = text
        self.url = url

    def raise_for_status(self):
        pass


def _install_fake_session(monkeypatch, replies, with_cache=False):
    """Make every requests.Session in _biomart answer with the next of ``replies``."""
    calls = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, params=None, timeout=None):
            calls.append({"url": url, "query": params["query"], "cache_disabled": getattr(self, "_off", None)})
            return _Response(replies[min(len(calls), len(replies)) - 1], url)

        if with_cache:

            @contextlib.contextmanager
            def cache_disabled(self):
                self._off = True
                yield
                self._off = False

    monkeypatch.setattr(_biomart.requests, "Session", FakeSession)
    monkeypatch.setattr("cellink.resources._utils.time.sleep", lambda s: None)
    return calls


def test_error_page_is_retried_over_https_and_tsv_parsed(monkeypatch):
    calls = _install_fake_session(monkeypatch, [ERROR_PAGE, "ENSG1\tA1BG\t1\nENSG2\t\t\n[success]\n"])
    df = query_biomart(
        ["ensembl_gene_id", "external_gene_name", "entrezgene_id"],
        filters={"chromosome_name": ["1", "X"]},
        genome_build="GRCh37",
    )
    assert len(calls) == 2  # the error page was not stored and replayed
    assert all(c["url"] == "https://grch37.ensembl.org/biomart/martservice" for c in calls)
    assert '<Filter name="chromosome_name" value="1,X"/>' in calls[0]["query"]
    assert 'completionStamp="1"' in calls[0]["query"]
    assert list(df.columns) == ["ensembl_gene_id", "external_gene_name", "entrezgene_id"]
    assert df["ensembl_gene_id"].tolist() == ["ENSG1", "ENSG2"]
    assert pd.isna(df.loc[1, "external_gene_name"]) and pd.isna(df.loc[1, "entrezgene_id"])


def test_cut_off_reply_raises_after_retries(monkeypatch):
    calls = _install_fake_session(monkeypatch, ["ENSG1\tA1BG\nENSG2\tA2"])
    with pytest.raises(RuntimeError, match="Incomplete BioMart reply"):
        query_biomart(["ensembl_gene_id", "external_gene_name"])
    assert len(calls) == 8


def test_query_error_is_not_retried(monkeypatch):
    calls = _install_fake_session(
        monkeypatch, ["Query ERROR: caught BioMart::Exception::Usage: Attribute x NOT FOUND\n"]
    )
    with pytest.raises(BiomartQueryError, match="NOT FOUND"):
        query_biomart(["x"])
    assert len(calls) == 1


def test_requests_cache_installed_by_others_is_bypassed(monkeypatch):
    calls = _install_fake_session(monkeypatch, ["[success]\n"], with_cache=True)
    df = query_biomart(["ensembl_gene_id"], genome_build="GRCh38")
    assert calls[0]["cache_disabled"] is True
    assert calls[0]["url"].startswith("https://www.ensembl.org/")
    assert df.empty and list(df.columns) == ["ensembl_gene_id"]


def test_host_override_and_invalid_build(monkeypatch):
    calls = _install_fake_session(monkeypatch, ["ENSG1\n[success]\n"])
    query_biomart(["ensembl_gene_id"], host="https://jun2026.archive.ensembl.org/")
    assert calls[0]["url"] == "https://jun2026.archive.ensembl.org/biomart/martservice"
    with pytest.raises(ValueError, match="genome_build"):
        query_biomart(["ensembl_gene_id"], genome_build="hg17")


def test_magma_entrez_pairs(monkeypatch):
    from cellink.tl.external._magma import _query_ensembl_entrez_pairs

    calls = _install_fake_session(monkeypatch, ["ENSG1\tA1BG\t1\nENSG2\tNOENTREZ\t\nENSG1\tA1BG\t1\n[success]\n"])
    pairs = _query_ensembl_entrez_pairs("GRCh38")
    assert pairs.to_dict("list") == {
        "ensembl_gene_id": ["ENSG1"],
        "external_gene_name": ["A1BG"],
        "entrezgene_id": [1],
    }
    assert pairs["entrezgene_id"].dtype.kind == "i"
    assert 'value="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y,MT"' in calls[0]["query"]


@pytest.mark.parametrize(("mode", "genes"), [("name", ["HGNC1", "EXT2"]), ("ensembl", ["ENSG1", "ENSG2"])])
def test_sldsc_annotation(monkeypatch, mode, genes):
    from cellink.tl.external._sldsc_utils import _fetch_ensembl_annotation

    _install_fake_session(
        monkeypatch,
        ["HGNC1\tEXT1\tENSG1\t1\t100\t200\tprotein_coding\n\tEXT2\tENSG2\tX\t300\t400\tlncRNA\n[success]\n"],
    )
    anno = _fetch_ensembl_annotation(genome_build="GRCh38", gene_identifier_mode=mode)
    assert list(anno.columns) == ["gene", "chrom", "start", "end", "gene_biotype"]
    assert anno["gene"].tolist() == genes
    assert anno["chrom"].tolist() == ["1", "X"]
    assert anno["start"].tolist() == [100, 300]


def test_sclinker_gene_coords(monkeypatch, tmp_path):
    from cellink.tl.external._sclinker_utils import _query_biomart_and_write_gene_coords

    _install_fake_session(
        monkeypatch,
        [
            "ENSG1\tCD19\t16\t100\t200\nENSG1\tCD19\t16\t90\t150\n"
            "ENSG2\t\t1\t5\t9\nENSG3\tPATCH\tHSCHR6_1\t1\t2\n[success]\n"
        ],
    )
    _query_biomart_and_write_gene_coords(tmp_path, genome_build="hg19")
    ensg = pd.read_csv(tmp_path / "gene_coord_ensembl_GRCh37.txt", sep=" ")
    hgnc = pd.read_csv(tmp_path / "gene_coord_hgnc_GRCh37.txt", sep=" ")
    assert ensg.to_dict("list") == {"GENE": ["ENSG1", "ENSG2"], "CHR": [16, 1], "START": [90, 5], "END": [200, 9]}
    assert hgnc.to_dict("list") == {"GENE": ["CD19"], "CHR": [16], "START": [90], "END": [200]}


def test_sclinker_gene_coords_leave_out_multi_locus_symbols(monkeypatch, tmp_path):
    # Y_RNA-like symbols name many Ensembl genes and are left out of the symbol table
    from cellink.tl.external._sclinker_utils import _query_biomart_and_write_gene_coords

    _install_fake_session(
        monkeypatch,
        [
            "ENSG1\tY_RNA\t1\t1000\t1100\nENSG2\tY_RNA\t1\t240000000\t240000100\n"
            "ENSG3\tY_RNA\t5\t50\t60\nENSG4\tCD19\t16\t100\t200\n[success]\n"
        ],
    )
    _query_biomart_and_write_gene_coords(tmp_path, genome_build="GRCh37")
    ensg = pd.read_csv(tmp_path / "gene_coord_ensembl_GRCh37.txt", sep=" ")
    hgnc = pd.read_csv(tmp_path / "gene_coord_hgnc_GRCh37.txt", sep=" ")
    assert list(ensg["GENE"]) == ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
    assert hgnc.to_dict("list") == {"GENE": ["CD19"], "CHR": [16], "START": [100], "END": [200]}


def test_helpers_do_not_use_pybiomart():
    import inspect

    from cellink.tl.external import _magma, _sclinker_utils, _sldsc_utils

    for module in (_magma, _sclinker_utils, _sldsc_utils):
        assert "pybiomart" not in inspect.getsource(module)
        assert "http://" not in inspect.getsource(module).replace("http://www.w3.org", "")
