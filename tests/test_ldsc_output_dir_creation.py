from __future__ import annotations

from cellink.tl.external._ldsc import _ensure_out_prefix_dir


def test_ensure_out_prefix_dir_creates_nested_directory(tmp_path):
    out_prefix = str(tmp_path / "a" / "b" / "c" / "out")
    assert not (tmp_path / "a" / "b" / "c").exists()
    _ensure_out_prefix_dir(out_prefix)
    assert (tmp_path / "a" / "b" / "c").exists()


def test_ensure_out_prefix_dir_noop_for_existing_directory(tmp_path):
    out_prefix = str(tmp_path / "out")
    _ensure_out_prefix_dir(out_prefix)
    _ensure_out_prefix_dir(out_prefix)


def test_ensure_out_prefix_dir_handles_bare_filename(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _ensure_out_prefix_dir("out")  
