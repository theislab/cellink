import os
import shlex
import sys

from cellink.tl._runner import _is_within
from cellink.tl.external._ldsc import LDSCRunner


def _runner(mode="singularity"):
    return LDSCRunner(
        config_dict={
            "execution_mode": mode,
            "singularity_image": "ldsc.sif",
            "docker_image": "ldsc",
            "ldsc_command": "ldsc.py",
            "make_annot_command": "make_annot.py",
            "munge_command": "munge_sumstats.py",
        }
    )


def test_is_within_compares_path_components(tmp_path):
    assert _is_within(str(tmp_path / "gwas" / "x"), str(tmp_path / "gwas"))
    assert _is_within(str(tmp_path / "gwas"), str(tmp_path / "gwas"))
    assert not _is_within(str(tmp_path / "gwas_data" / "x"), str(tmp_path / "gwas"))


def test_local_command_runs_without_shell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = _runner("local")
    runner.run_command(shlex.join(["echo", "gwas$(touch pwned).sumstats.gz; touch pwned2"]))
    assert not (tmp_path / "pwned").exists() and not (tmp_path / "pwned2").exists()

    (tmp_path / "My GWAS").mkdir()
    runner.run_command(shlex.join([sys.executable, "-c", "import sys; open(sys.argv[1], 'w')", "My GWAS/height.txt"]))
    assert (tmp_path / "My GWAS" / "height.txt").is_file()


def test_sibling_directory_gets_its_own_bind(tmp_path, monkeypatch):
    work, sibling = tmp_path / "gwas", tmp_path / "gwas_data"
    work.mkdir()
    sibling.mkdir()
    (sibling / "height.sumstats.gz").write_text("x")
    monkeypatch.chdir(work)
    sumstats = str(sibling / "height.sumstats.gz")
    runner = _runner()

    full = runner._build_container_command(f"ldsc.py --h2 {sumstats} --out out", [sumstats, os.getcwd()])
    args = shlex.split(full)
    assert f"{sibling}:/external_2" in args or any(a.startswith(f"{sibling}:/external_") for a in args)
    h2 = args[args.index("--h2") + 1]
    assert h2.startswith("/external_") and h2.endswith("/height.sumstats.gz")

    # host paths inside --flag=value tokens (SAIGE-QTL style) are rewritten at path boundaries only
    full = runner._build_container_command(f"Rscript step2.R --bedFile={sibling}/g.bed", [str(sibling / "g.bed")])
    bed = [a for a in shlex.split(full) if a.startswith("--bedFile=")][0]
    assert bed.startswith("--bedFile=/external_") and "data_data" not in bed


def test_container_command_quotes_paths_with_spaces(tmp_path, monkeypatch):
    work = tmp_path / "my work"
    work.mkdir()
    (work / "x.sumstats.gz").write_text("x")
    monkeypatch.chdir(work)
    for mode, flag in [("singularity", "-B"), ("docker", "-v")]:
        args = shlex.split(_runner(mode)._build_container_command("ldsc.py --h2 x.sumstats.gz --out out", []))
        assert args[args.index(flag) + 1] == f"{work}:/data"
        assert args[-5:] == ["ldsc.py", "--h2", "/data/x.sumstats.gz", "--out", "/data/out"]
