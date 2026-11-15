import gzip
import http.server
import io
import logging
import socket
import tarfile
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from cellink.resources import _gwas_prs_qtl, _ld, _utils

# --------------------------------------------------------------------------- helpers


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


# --------------------------------------------------------------------------- summary statistics liftover


def test_liftover_unknown_build_raises_clear_error():
    df = pd.DataFrame({"chromosome": ["1"], "base_pair_location": [1000]})
    with pytest.raises(ValueError, match="genome build of the summary statistics is unknown"):
        _gwas_prs_qtl.liftover_gwas_summary_stats(df, "unknown", "GRCh38")


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


def test_download_log_spells_verifying(tmp_path, caplog):
    path = tmp_path / "f.txt"
    path.write_text("abc")
    with caplog.at_level(logging.INFO):
        _utils._download_file("https://example.invalid/f.txt", path, checksum=_utils._sha256sum(path))
    assert "Verifying checksum" in caplog.text and "Veryifying" not in caplog.text


# --------------------------------------------------------------------------- caches


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


def test_interrupted_extraction_is_redone(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    archive = _tgz(target / "data.tgz", {"pkg/a.txt": b"full a", "pkg/b.txt": b"full b"})
    (target / "a.txt").write_bytes(b"part")  # left by an interrupted extraction, without the completion marker
    _ld._extract_or_refresh(archive, target)
    assert (target / "a.txt").read_bytes() == b"full a"
    assert (target / "b.txt").read_bytes() == b"full b"
    assert sorted(p.name for p in target.iterdir()) == [".data.tgz.extracted", "a.txt", "b.txt", "data.tgz"]


def test_finished_extraction_is_not_repeated(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    archive = _tgz(target / "data.tgz", {"a.txt": b"a", "b.txt": b"b"})
    _ld._extract_or_refresh(archive, target)
    assert (target / "a.txt").read_bytes() == b"a"
    (target / "a.txt").write_bytes(b"changed")
    _ld._extract_or_refresh(archive, target)
    assert (target / "a.txt").read_bytes() == b"changed"
    _ld._extract_or_refresh(archive, target, refresh=True)
    assert (target / "a.txt").read_bytes() == b"a"


def test_failed_extraction_leaves_no_marker(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    archive = _tgz(target / "evil.tgz", {"ok.txt": b"x", "../escaped.txt": b"x"})
    with pytest.raises(tarfile.FilterError):
        _ld._extract_or_refresh(archive, target)
    assert sorted(p.name for p in target.iterdir()) == ["evil.tgz"]
