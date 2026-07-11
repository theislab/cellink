from pathlib import Path

import pytest

DATA = Path("tests/data")


@pytest.mark.slow
def test_pgen_cli_main(tmp_path, monkeypatch):
    pytest.importorskip("pgenlib")

    from cellink.cli.pgen import main

    out_path = tmp_path / "cli_out.zarr"
    argv = [
        "cellink-pgen",
        str(DATA / "simulated_genotype_calls.pgen"),
        "-o",
        str(out_path),
        "--chunk-samples",
        "50",
        "--chunk-variants",
        "200",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert main() == 0

    from cellink.io import read_pgen_zarr

    adata = read_pgen_zarr(str(out_path))
    assert adata.shape == (100, 1000)
