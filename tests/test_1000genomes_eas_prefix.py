from __future__ import annotations

from pathlib import Path

import yaml

import cellink


def _load_config():
    config_path = Path(cellink.__file__).parent / "resources" / "config" / "1000genomes.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_plink_files_prefix_is_population_specific():
    config = _load_config()
    assert config["plink_files"]["EUR"]["prefix"] == "1000G.EUR.QC."
    assert config["plink_files"]["EAS"]["prefix"] == "1000G.EAS.QC."


def test_ld_weights_prefix_is_population_specific():
    config = _load_config()
    assert config["ld_weights"]["EUR"]["prefix"] == "weights.hm3_noMHC."
    assert config["ld_weights"]["EAS"]["prefix"] == "weights.EAS.hm3_noMHC."


def test_ld_scores_prefix_genuinely_is_population_independent():
    config = _load_config()
    assert config["ld_scores"]["prefix"] == "baselineLD."


def test_get_1000genomes_ld_weights_dataframe_path_does_not_raise_nameerror(tmp_path, monkeypatch):
    import pandas as pd

    from cellink.resources import _ld

    data_home = tmp_path / "cellink_data"
    weights_dir = data_home / "1000genomes_ld_weights_EUR"
    weights_dir.mkdir(parents=True)
    for chrom in range(1, 23):
        pd.DataFrame({"SNP": [f"rs{chrom}"], "L2": [1.0]}).to_csv(
            weights_dir / f"weights.hm3_noMHC.{chrom}.l2.ldscore.gz", sep="\t", index=False, compression="gzip"
        )

    monkeypatch.setattr(_ld, "_download_file", lambda *a, **k: None)
    monkeypatch.setattr(_ld, "_extract_or_refresh", lambda *a, **k: None)

    result = _ld.get_1000genomes_ld_weights(
        config_path=str(Path(cellink.__file__).parent / "resources" / "config" / "1000genomes.yaml"),
        population="EUR",
        data_home=str(data_home),
        return_path=False,
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 22
