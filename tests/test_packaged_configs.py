from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from cellink import resources
from cellink._config import PACKAGED_CONFIG_DIRS, load_config, packaged_configs, resolve_config_path
from cellink.tl import _annotate_snps_genotype_data as annotate
from cellink.tl.external import _ldsc, _livi, _magma, _saigeqtl

FUNCTIONS_WITH_CONFIG_DEFAULTS = [
    (resources.get_1000genomes, "config_path"),
    (resources.get_1000genomes_grch38, "config_path"),
    (resources.get_onek1k, "config_path"),
    (resources.get_dummy_onek1k, "config_path"),
    (resources.get_1000genomes_ld_scores, "config_path"),
    (resources.get_1000genomes_ld_weights, "config_path"),
    (resources.get_1000genomes_plink_files, "config_path"),
    (resources.get_1000genomes_frq, "config_path"),
    (resources.get_1000genomes_hapmap3, "config_path"),
    (resources.get_1000genomes_merge_alleles, "config_path"),
    (annotate.download_favor, "config_file"),
    (annotate.run_favor, "config_file"),
    (annotate.run_vep, "config_file"),
    (_magma.load_magma_config, "config_file"),
    (_magma.download_magma_references, "config_file"),
    (_magma.prepare_magma_inputs, "config_file"),
    (_magma.run_magma_pipeline, "config_file"),
    (_ldsc.LDSCRunner.__init__, "config_path"),
    (_ldsc.configure_ldsc_runner, "config_path"),
    (_saigeqtl.SAIGEQTLRunner.__init__, "config_path"),
    (_saigeqtl.configure_saigeqtl_runner, "config_path"),
    (_livi.configure_livi_runner, "config_path"),
]


@pytest.mark.parametrize(("func", "param"), FUNCTIONS_WITH_CONFIG_DEFAULTS, ids=lambda x: getattr(x, "__qualname__", x))
def test_config_default_resolves_to_a_shipped_config(func, param, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    default = inspect.signature(func).parameters[param].default
    resolved = resolve_config_path(default)
    assert resolved.parent in PACKAGED_CONFIG_DIRS
    assert isinstance(yaml.safe_load(resolved.read_text()), dict)


@pytest.mark.parametrize(
    ("func", "param"),
    [*FUNCTIONS_WITH_CONFIG_DEFAULTS, (annotate.load_favor_urls, "config_file")],
    ids=lambda x: getattr(x, "__qualname__", x),
)
def test_every_config_function_also_takes_a_config_dict(func, param):
    params = inspect.signature(func).parameters
    assert param in params
    assert params["config_dict"].default is None


def test_config_dict_takes_precedence_over_the_config_path():
    favor = {"favor": {"essential": {1: "https://example.org/chr1"}}}
    assert load_config("does_not_exist", favor) is favor
    urls = annotate.load_favor_urls("does_not_exist", "essential", config_dict=favor)
    assert urls.to_dict("records") == [{"chr": "1", "URL": "https://example.org/chr1"}]
    assert _magma.load_magma_config("does_not_exist", config_dict={"gene_loc": {}}) == {"gene_loc": {}}


def test_livi_runner_from_config_dict(tmp_path):
    runner = _livi.configure_livi_runner(config_dict={"livi_root": str(tmp_path), "device": "cpu"})
    assert runner.livi_root == tmp_path.resolve()
    assert runner.device == "cpu"


def test_shipped_config_names_are_unique():
    names = [path.stem for config_dir in PACKAGED_CONFIG_DIRS for path in config_dir.glob("*.yaml")]
    assert len(names) == len(set(names))
    assert {"1000genomes", "magma", "favor", "vep", "ldsc_docker", "saigeqtl_singularity", "livi"} <= set(names)


def test_name_prefers_shipped_config_over_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "vep.yaml").write_text("vep_command: my_vep\n")

    assert resolve_config_path("vep") == packaged_configs()["vep"]
    assert resolve_config_path("vep.yaml") == Path("vep.yaml")
    assert load_config("./vep.yaml") == {"vep_command": "my_vep"}


def test_unknown_name_falls_back_to_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "my_ldsc.yaml").write_text("execution_mode: local\n")

    assert resolve_config_path("my_ldsc") == Path("my_ldsc.yaml")


def test_missing_config_raises_and_never_falls_back_to_a_shipped_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Shipped configs: .*ldsc_docker"):
        resolve_config_path("ldsc_dockr")
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_config_path("../configs/vep.yaml")


def test_ldsc_runner_defaults_match_the_previous_hard_coded_config():
    assert _ldsc.LDSCRunner().config == {
        "execution_mode": "local",
        "docker_image": "zijingliu/ldsc",
        "singularity_image": None,
        "ldsc_command": "ldsc.py",
        "make_annot_command": "make_annot.py",
        "munge_command": "munge_sumstats.py",
    }
    assert _ldsc.LDSCRunner(config_path=None).config == _ldsc.LDSCRunner().config


def test_saigeqtl_runner_defaults_match_the_previous_hard_coded_config():
    assert _saigeqtl.SAIGEQTLRunner().config == {
        "execution_mode": "local",
        "rscript_path": "Rscript",
        "docker_image": "wzhou88/saigeqtl",
        "singularity_image": "docker://wzhou88/saigeqtl",
        "step1_script": "step1_fitNULLGLMM_qtl.R",
        "step2_script": "step2_tests_qtl.R",
        "step3_script": "step3_gene_pvalue_qtl.R",
        "makegroup_script": "makeGroupFile.R",
    }


@pytest.mark.parametrize(
    ("configure", "name", "mode"),
    [
        (_ldsc.configure_ldsc_runner, "ldsc_docker", "docker"),
        (_ldsc.configure_ldsc_runner, "ldsc_singularity", "singularity"),
        (_saigeqtl.configure_saigeqtl_runner, "saigeqtl_docker", "docker"),
        (_saigeqtl.configure_saigeqtl_runner, "saigeqtl_singularity", "singularity"),
    ],
)
def test_runner_container_configs_by_name(configure, name, mode):
    assert configure(name).execution_mode == mode


def test_runner_with_missing_config_path_raises_instead_of_using_defaults(tmp_path):
    with pytest.raises(FileNotFoundError):
        _ldsc.LDSCRunner(str(tmp_path / "typo" / "ldsc.yaml"))


def test_livi_runner_takes_unset_arguments_from_config(tmp_path):
    livi_root = tmp_path / "LIVI"
    livi_root.mkdir()
    config = tmp_path / "livi.yaml"
    config.write_text(f"livi_root: {livi_root}\nexecution_mode: subprocess\ndevice: cpu\n")

    runner = _livi.configure_livi_runner(config_path=config, device="cuda")

    assert runner.livi_root == livi_root.resolve()
    assert runner.execution_mode == "subprocess"
    assert runner.python_executable == "python"
    assert runner.device == "cuda"


def test_livi_runner_shipped_defaults_and_missing_root(tmp_path):
    runner = _livi.configure_livi_runner(tmp_path)
    assert (runner.execution_mode, runner.python_executable, runner.device) == ("python_api", "python", "auto")

    with pytest.raises(ValueError, match="livi_root is required"):
        _livi.configure_livi_runner()
