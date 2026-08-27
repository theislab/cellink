from __future__ import annotations

import os

from cellink.tl.external._ldsc import LDSCRunner


def _make_runner():
    return LDSCRunner(
        config_dict={
            "execution_mode": "singularity",
            "singularity_image": "dummy.sif",
            "ldsc_command": "ldsc.py",
            "make_annot_command": "make_annot.py",
            "munge_command": "munge_sumstats.py",
        }
    )


def test_ref_ld_chr_comma_separated_paths_both_rewritten():
    runner = _make_runner()
    cmd = "ldsc.py --h2 x.sumstats.gz --ref-ld-chr baselineLD/extracted/baselineLD.,custom/my_annot. --w-ld-chr weights. --out out"
    full = runner._build_container_command(cmd, [os.getcwd()])

    assert "--ref-ld-chr /data/baselineLD/extracted/baselineLD.,/data/custom/my_annot." in full


def test_single_path_prefix_token_still_rewritten_normally():
    runner = _make_runner()
    cmd = "ldsc.py --h2 x.sumstats.gz --ref-ld-chr baselineLD/extracted/baselineLD. --w-ld-chr weights. --out out"
    full = runner._build_container_command(cmd, [os.getcwd()])

    assert "--ref-ld-chr /data/baselineLD/extracted/baselineLD." in full


def test_three_way_comma_separated_paths_all_rewritten():
    runner = _make_runner()
    cmd = "ldsc.py --h2 x.sumstats.gz --ref-ld-chr a/one.,b/two.,c/three. --out out"
    full = runner._build_container_command(cmd, [os.getcwd()])
    assert "--ref-ld-chr /data/a/one.,/data/b/two.,/data/c/three." in full
