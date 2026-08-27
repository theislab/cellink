from __future__ import annotations

import pandas as pd

from cellink.resources._ld import get_1000genomes_merge_alleles


def test_get_1000genomes_merge_alleles_builds_real_3col_format(tmp_path, monkeypatch):
    plink_dir = tmp_path / "plink"
    plink_dir.mkdir()
    bim1 = pd.DataFrame(
        {
            "CHR": ["1", "1", "1"],
            "SNP": ["rs1", "rs2", "rs3"],
            "CM": [0.0, 0.0, 0.0],
            "BP": [100, 200, 300],
            "A1": ["A", "C", "G"],
            "A2": ["G", "T", "A"],
        }
    )
    bim1.to_csv(plink_dir / "1000G.EUR.QC.1.bim", sep="\t", header=False, index=False)

    hapmap3_dir = tmp_path / "hapmap3"
    hapmap3_dir.mkdir()
    (hapmap3_dir / "hm3_no_MHC.list.txt").write_text("rs1\nrs2\n")  # rs3 excluded

    def fake_plink(*args, **kwargs):
        return plink_dir, "1000G.EUR.QC."

    def fake_hapmap3(*args, **kwargs):
        return hapmap3_dir / "hm3_no_MHC.list.txt"

    monkeypatch.setattr("cellink.resources._ld.get_1000genomes_plink_files", fake_plink)
    monkeypatch.setattr("cellink.resources._ld.get_1000genomes_hapmap3", fake_hapmap3)

    dest = get_1000genomes_merge_alleles(data_home=tmp_path / "out", chromosomes=[1])

    df = pd.read_csv(dest, sep="\t")
    assert list(df.columns) == ["SNP", "A1", "A2"]
    assert set(df["SNP"]) == {"rs1", "rs2"}  # rs3 correctly excluded (not in HapMap3)
    assert df.set_index("SNP").loc["rs1"].tolist() == ["A", "G"]
