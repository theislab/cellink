import gzip
import http.server
import importlib
import io
import logging
import socket
import sys
import tarfile
import threading
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellink.resources import _datasets_utils, _gwas_prs_qtl, _ld, _utils

# --------------------------------------------------------------------------- helpers


class _FakeChain:
    """Stand-in for a ``liftover`` chain: ``chain[chrom][pos] -> [(chrom, pos, strand)]``."""

    def __init__(self, table):
        self.table = table  # {(chrom, pos): (new_chrom, new_pos)}

    def __getitem__(self, chrom):
        chrom = str(chrom).removeprefix("chr")
        return _FakeChrom(self.table, chrom)


class _FakeChrom:
    def __init__(self, table, chrom):
        self.table, self.chrom = table, chrom

    def __getitem__(self, pos):
        hit = self.table.get((self.chrom, int(pos)))
        return [(hit[0], hit[1], "+")] if hit else []


@pytest.fixture
def fake_liftover(monkeypatch):
    """Install a fake ``liftover`` module and record the builds that were requested."""
    calls = []
    table = {
        ("22", 19963748): ("chr22", 19951271),  # rs4680, hg38 -> hg19
        ("9", 45950000): ("chr21", 9996945),  # lifts to another chromosome
        ("1", 1000): ("chr1", 2000),
        ("1", 1100): ("chr1", 2100),
    }

    def get_lifter(src, tgt, one_based=True, cache=None):
        calls.append((src, tgt))
        return _FakeChain(table)

    monkeypatch.setitem(sys.modules, "liftover", types.SimpleNamespace(get_lifter=get_lifter))
    _datasets_utils._get_hg38_to_hg19_converter.cache_clear()
    yield calls
    _datasets_utils._get_hg38_to_hg19_converter.cache_clear()


class _Handler(http.server.BaseHTTPRequestHandler):
    body = b""
    announce = None  # Content-Length to send, defaults to len(body)
    stall = False
    hits = 0

    def do_GET(self):
        type(self).hits += 1
        if self.stall:
            time.sleep(5)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(self.announce if self.announce is not None else len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *args):
        pass


@pytest.fixture
def http_server():
    servers = []

    def start(body=b"", announce=None, stall=False):
        handler = type("H", (_Handler,), {"body": body, "announce": announce, "stall": stall, "hits": 0})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/file", handler

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def _no_part_files(directory: Path) -> bool:
    return not list(directory.glob("*.part"))


# --------------------------------------------------------------------------- OneK1K liftover and gPCs


def test_datasets_utils_import_does_not_load_liftover(monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise AssertionError("the liftover chain must not be loaded at import time")

    monkeypatch.setitem(sys.modules, "liftover", types.SimpleNamespace(get_lifter=fail))
    with caplog.at_level(logging.WARNING):
        importlib.reload(_datasets_utils)
    assert "CELLINK_LIFTOVER_CACHE" not in caplog.text


def test_try_liftover_lifts_grch38_to_hg19(fake_liftover):
    assert _datasets_utils.try_liftover(pd.Series({"chrom": "22", "pos": 19963748})) == 19951271
    assert fake_liftover == [("hg38", "hg19")]
    # a position that maps to another chromosome has no hg19 position on this chromosome
    assert np.isnan(_datasets_utils.try_liftover(pd.Series({"chrom": "9", "pos": 45950000})))
    assert np.isnan(_datasets_utils.try_liftover(pd.Series({"chrom": "22", "pos": 1})))
    assert fake_liftover == [("hg38", "hg19")]  # loaded once


def test_get_onek1k_names_gpcs_from_one_and_lifts_to_hg19(tmp_path, monkeypatch, fake_liftover):
    import anndata as ad

    import cellink as cl
    from cellink.resources import _datasets

    data = tmp_path / "onek1k"
    (data / "OneK1K.noGP.vcz").mkdir(parents=True)  # skips the vcf2zarr/PLINK preprocessing
    donors = ["OneK1K_1", "OneK1K_2", "OneK1K_3"]
    var = pd.DataFrame(
        {"contig": [0, 0], "chrom": ["22", "22"], "pos": [19963748, 5], "a0": ["G", "A"], "a1": ["A", "C"]},
        index=["22_19963748_G_A", "22_5_A_C"],
    )
    gdata = ad.AnnData(
        X=np.zeros((3, 2), dtype="float32"), obs=pd.DataFrame(index=pd.Index(donors, name="donor_id")), var=var
    )
    monkeypatch.setattr(cl.io, "read_sgkit_zarr", lambda path: gdata.copy())

    (data / "pcdir").mkdir()
    rows = [f"0 {d} " + " ".join(str(10 * i + k) for k in range(1, 4)) for i, d in enumerate(donors)]
    (data / "pcdir" / "OneK1K.noGP.filtered.pruned.eigenvec").write_text("\n".join(rows) + "\n")
    (data / "kinship").mkdir()
    (data / "kinship" / "OneK1K.noGP.filtered.pruned.rel").write_text("1\t0\t0\n0\t1\t0\n0\t0\t1\n")
    (data / "kinship" / "OneK1K.noGP.filtered.pruned.rel.id").write_text("".join(f"0\t{d}\n" for d in donors))
    obs = pd.DataFrame(
        {"donor_id": ["x_1", "x_2", "x_3"], "sex": ["male", "female", "female"], "age": ["30", "40", "50"]},
        index=["c1", "c2", "c3"],
    )
    ad.AnnData(X=np.zeros((3, 1), dtype="float32"), obs=obs).write_h5ad(data / "onek1k_cellxgene.h5ad")

    dd = _datasets.get_onek1k(config_dict={"remote_files": []}, data_home=tmp_path)

    gpcs = dd.G.obsm["gPCs"]
    assert list(gpcs.columns) == ["gPCs_1", "gPCs_2", "gPCs_3"]
    assert gpcs.loc["OneK1K_2", "gPCs_1"] == 11  # PC1 is the first eigenvector column
    assert list(dd.G.var["pos_hg19"].astype("float")[:1]) == [19951271]
    assert pd.isna(dd.G.var["pos_hg19"].iloc[1])
    assert dd.G.var["id_hg19"].iloc[0] == "22_19951271_G_A"


# --------------------------------------------------------------------------- summary statistics liftover


def test_liftover_takes_chromosome_from_lift_and_returns_int64(fake_liftover):
    df = pd.DataFrame({"chromosome": ["9", "1", "1"], "base_pair_location": [45950000, 1000, 5], "beta": [1, 2, 3]})
    out = _gwas_prs_qtl.liftover_gwas_summary_stats(df, "GRCh37", "GRCh38", drop_failed=False)
    assert list(out["chromosome"]) == ["21", "1", "1"]
    assert out["base_pair_location"].dtype == "Int64"
    assert list(out["base_pair_location"][:2]) == [9996945, 2000] and pd.isna(out["base_pair_location"][2])
    assert fake_liftover == [("hg19", "hg38")]


def test_liftover_float_chromosomes_and_na_rows(fake_liftover):
    df = pd.DataFrame(
        {
            "chromosome": [1.0, 1.0, np.nan, 9.0],  # float, as pandas reads a column with NAs
            "base_pair_location": [1000, np.nan, 1100, 45950000],
        }
    )
    out = _gwas_prs_qtl.liftover_gwas_summary_stats(df, "hg19", "hg38")
    assert out["chromosome"].dtype == "Int64"
    assert list(out["chromosome"]) == [1, 21]
    assert list(out["base_pair_location"]) == [2000, 9996945]
    assert out["base_pair_location"].dtype == "Int64"


def test_liftover_keeps_chr_prefix_style(fake_liftover):
    df = pd.DataFrame({"chromosome": ["chr1", "chr9"], "base_pair_location": [1100, 45950000]})
    out = _gwas_prs_qtl.liftover_gwas_summary_stats(df, "GRCh37", "GRCh38")
    assert list(out["chromosome"]) == ["chr1", "chr21"]


def test_liftover_unknown_build_raises_clear_error():
    df = pd.DataFrame({"chromosome": ["1"], "base_pair_location": [1000]})
    with pytest.raises(ValueError, match="genome build of the summary statistics is unknown"):
        _gwas_prs_qtl.liftover_gwas_summary_stats(df, "unknown", "GRCh38")


def _sumstats_gz(path: Path, rows: str = "chromosome\tbase_pair_location\tbeta\n1\t1000\t0.1\n") -> Path:
    with gzip.open(path, "wt") as fh:
        fh.write(rows)
    return path


def test_summary_stats_translate_with_undetected_build_raises(tmp_path, monkeypatch):
    src = _sumstats_gz(tmp_path / "study.tsv.gz")
    monkeypatch.setattr(
        _gwas_prs_qtl,
        "_find_gwas_catalog_summary_stats_file",
        lambda accession, build, params: (src.as_uri(), "study.tsv.gz", "unknown"),
    )
    with pytest.raises(ValueError, match="Could not detect the genome build"):
        _gwas_prs_qtl.get_gwas_catalog_study_summary_stats(
            "GCST1", translate_to_build="GRCh37", data_home=tmp_path / "home"
        )


# --------------------------------------------------------------------------- downloads and extraction


def test_download_file_writes_complete_file(tmp_path, http_server):
    url, handler = http_server(body=b"x" * 3000)
    _utils._download_file(url, tmp_path / "f.bin")
    assert (tmp_path / "f.bin").read_bytes() == b"x" * 3000
    _utils._download_file(url, tmp_path / "f.bin")  # existing file is reused
    assert handler.hits == 1
    _utils._download_file(url, tmp_path / "f.bin", refresh=True)
    assert handler.hits == 2
    assert _no_part_files(tmp_path)


def test_download_file_discards_truncated_download(tmp_path, http_server):
    url, _ = http_server(body=b"x" * 100, announce=1000)  # server closes after 100 of 1000 bytes
    with pytest.raises(OSError):
        _utils._download_file(url, tmp_path / "f.gz")
    assert not (tmp_path / "f.gz").exists()
    assert _no_part_files(tmp_path)


def test_download_file_times_out(tmp_path, http_server):
    url, _ = http_server(stall=True)
    start = time.monotonic()
    with pytest.raises((TimeoutError, socket.timeout, OSError)):
        _utils._download_file(url, tmp_path / "f.gz", timeout=0.5)
    assert time.monotonic() - start < 4
    assert not (tmp_path / "f.gz").exists()


def test_download_file_checksum_mismatch_keeps_nothing(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("payload")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        _utils._download_file(src.as_uri(), tmp_path / "out" / "f.txt", checksum="0" * 64)
    assert not (tmp_path / "out" / "f.txt").exists()
    _utils._download_file(src.as_uri(), tmp_path / "out" / "f.txt", checksum=_utils._sha256sum(src))
    assert (tmp_path / "out" / "f.txt").read_text() == "payload"
    assert _no_part_files(tmp_path / "out")


def _tgz(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_ld_scores_refresh_keeps_archive_and_reextracts(tmp_path):
    archive = _tgz(
        tmp_path / "ld.tgz",
        {
            "ld/baselineLD.1.annot.gz": gzip.compress(b"SNP\tA\nrs1\t1\n"),
            "ld/baselineLD.1.l2.ldscore.gz": gzip.compress(b"SNP\tL2\nrs1\t1.0\n"),
        },
    )
    config = {"ld_scores": {"prefix": "baselineLD.", "EUR": {"filename": "ld.tgz", "url": archive.as_uri()}}}
    home = tmp_path / "home"
    annot, ldscores, prefix = _ld.get_1000genomes_ld_scores(config_dict=config, data_home=home)
    assert list(ldscores["SNP"]) == ["rs1"]
    annot, ldscores, prefix = _ld.get_1000genomes_ld_scores(config_dict=config, data_home=home, refresh=True)
    assert list(ldscores["SNP"]) == ["rs1"]
    assert (home / "1000genomes_ld_scores_EUR" / "ld.tgz").exists()


def test_extract_rejects_members_outside_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    archive = _tgz(target / "evil.tgz", {"../escaped.txt": b"x"})
    with pytest.raises(tarfile.FilterError):
        _ld._extract_or_refresh(archive, target)
    assert not (tmp_path / "escaped.txt").exists()


def test_hapmap3_honours_refresh(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(_ld, "_download_file", lambda url, dest, checksum=None, refresh=False: seen.append(refresh))
    config = {"hapmap3": {"filename": "hm3.txt", "url": "https://example.org/hm3.txt"}}
    _ld.get_1000genomes_hapmap3(config_dict=config, data_home=tmp_path)
    _ld.get_1000genomes_hapmap3(config_dict=config, data_home=tmp_path, refresh=True)
    assert seen == [False, True]


def test_download_log_spells_verifying(tmp_path, caplog):
    path = tmp_path / "f.txt"
    path.write_text("abc")
    with caplog.at_level(logging.INFO):
        _utils._download_file("https://example.invalid/f.txt", path, checksum=_utils._sha256sum(path))
    assert "Verifying checksum" in caplog.text and "Veryifying" not in caplog.text


# --------------------------------------------------------------------------- caches


def test_catalog_caches_are_keyed_on_query(tmp_path, monkeypatch):
    fetched = []

    def fake_fetch(url, params=None, paginate=True, max_pages=None):
        fetched.append((url, dict(params or {}), max_pages))
        return [{"query": (params or {}).get("q", "all")}]

    monkeypatch.setattr(_gwas_prs_qtl, "_fetch", fake_fetch)
    for getter in (_gwas_prs_qtl.get_gwas_catalog_studies, _gwas_prs_qtl.get_pgs_catalog_scores):
        fetched.clear()
        assert list(getter(data_home=tmp_path, q="asthma")["query"]) == ["asthma"]
        assert list(getter(data_home=tmp_path, q="height")["query"]) == ["height"]
        assert list(getter(data_home=tmp_path, q="asthma")["query"]) == ["asthma"]
        assert len(fetched) == 2  # the repeated query came from the cache
        getter(data_home=tmp_path, q="asthma", max_pages=1)
        assert len(fetched) == 3


def test_merge_alleles_file_depends_on_chromosomes(tmp_path, monkeypatch):
    plink_dir = tmp_path / "plink"
    plink_dir.mkdir()
    for chrom, snp in ((1, "rs1"), (2, "rs2")):
        (plink_dir / f"1000G.EUR.QC.{chrom}.bim").write_text(f"{chrom}\t{snp}\t0\t100\tA\tG\n")
    (tmp_path / "hm3.txt").write_text("rs1\nrs2\n")
    monkeypatch.setattr(_ld, "get_1000genomes_plink_files", lambda **kw: (plink_dir, "1000G.EUR.QC."))
    monkeypatch.setattr(_ld, "get_1000genomes_hapmap3", lambda **kw: tmp_path / "hm3.txt")

    first = _ld.get_1000genomes_merge_alleles(data_home=tmp_path / "out", chromosomes=[1])
    second = _ld.get_1000genomes_merge_alleles(data_home=tmp_path / "out", chromosomes=[2])
    assert first != second
    assert list(pd.read_csv(first, sep="\t")["SNP"]) == ["rs1"]
    assert list(pd.read_csv(second, sep="\t")["SNP"]) == ["rs2"]


def test_merge_plink_chromosomes_checks_full_prefix(tmp_path, monkeypatch):
    import subprocess

    commands = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    (tmp_path / "ref.bed").write_text("")  # not the output of prefix "ref.v2"
    out = _ld.merge_1000g_plink_chromosomes(tmp_path, "1000G.", tmp_path / "ref.v2", chromosomes=[1, 2])
    assert out == tmp_path / "ref.v2"
    assert len(commands) == 1 and "--out" in commands[0]

    (tmp_path / "ref.v2.bed").write_text("")
    _ld.merge_1000g_plink_chromosomes(tmp_path, "1000G.", tmp_path / "ref.v2", chromosomes=[1, 2])
    assert len(commands) == 1  # existing merged output is reused


def test_summary_stats_download_is_reused_per_build(tmp_path, monkeypatch):
    files = {
        "GRCh38": _sumstats_gz(tmp_path / "GCST1.h.tsv.gz"),
        "GRCh37": _sumstats_gz(tmp_path / "GCST1_buildGRCh37.tsv.gz"),
    }
    lookups = []

    def fake_find(accession, build, params):
        lookups.append(build)
        resolved = build or "GRCh38"
        return files[resolved].as_uri(), files[resolved].name, resolved

    monkeypatch.setattr(_gwas_prs_qtl, "_find_gwas_catalog_summary_stats_file", fake_find)
    get = _gwas_prs_qtl.get_gwas_catalog_study_summary_stats
    home = tmp_path / "home"

    p38 = get("GCST1", return_path=True, data_home=home)
    assert get("GCST1", return_path=True, data_home=home) == p38
    assert lookups == [None]  # the second call did not list the FTP directory

    p37 = get("GCST1", return_path=True, genome_build="hg19", data_home=home)
    assert p37 != p38 and p37.name == "GCST1_buildGRCh37.tsv.gz" and p38.exists()
    df = get("GCST1", genome_build="GRCh37", data_home=home)
    assert list(df["beta"]) == [0.1]
    assert lookups == [None, "GRCh37"]

    get("GCST1", return_path=True, data_home=home, refresh=True)
    assert lookups == [None, "GRCh37", None]
    assert _no_part_files(p38.parent)
