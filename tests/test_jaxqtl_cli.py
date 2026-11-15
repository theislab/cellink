import gzip
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cellink.tl.external._jaxqtl as jx
from cellink._core.donordata import DonorData
from cellink.tl.external._jaxqtl import read_jaxqtl_results, run_jaxqtl


@pytest.fixture
def dd(adata, gdata):
    gdata.obs["donor_id"] = gdata.obs.index
    gdata.obs["sex"] = np.resize([1, 2], gdata.n_obs)
    gdata.obs["age"] = np.arange(gdata.n_obs) + 30
    # real gene annotations are float when any coordinate is missing
    adata.var["start"] = adata.var["start"].astype(float)
    adata.var["end"] = adata.var["end"].astype(float)
    return DonorData(G=gdata, C=adata)


def _commands(cmd):
    return [shlex.split(c) for c in cmd.split(" && ")]


def _values(args, flag):
    """Values following ``flag`` up to the next option."""
    start = args.index(flag) + 1
    stop = next((i for i in range(start, len(args)) if args[i].startswith("--")), len(args))
    return args[start:stop]


def test_default_commands(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    pcs, scan = _commands(run_jaxqtl(dd, prefix="jx", run=False))
    assert pcs == [
        "jaxqtl",
        "compute-pcs",
        "--pheno",
        "jx_phenotype.bed.gz",
        "--covar",
        "jx_donor_features.tsv",
        "--num-pcs",
        "2",
        "--normalization",
        "library-size",
        "--transform",
        "log1p",
        "--out",
        "jx_donor_features_pcs.tsv",
    ]
    assert scan[:8] == ["jaxqtl", "cis", "--bfile", "jx", "--pheno", "jx_phenotype.bed.gz", "--covar"] + [
        "jx_donor_features_pcs.tsv"
    ]
    assert scan[8:] == [
        "--model",
        "nb",
        "--test",
        "score",
        "--nperm",
        "1000",
        "--window",
        "500000",
        "--normalize-covar",
        "--set-offset-from-libsize",
        "--out",
        "jx",
    ]
    assert all((tmp_path / f"jx.{ext}").is_file() for ext in ("bed", "bim", "fam"))

    # jaxqtl rejects float gene coordinates ("chrom:16584954.0-17615693.0")
    with gzip.open(tmp_path / "jx_phenotype.bed.gz", "rt") as f:
        header, first = f.readline(), f.readline()
    assert header.startswith("#chr\tstart\tend\tGeneid")
    start, end = first.split("\t")[1:3]
    assert start.isdigit() and end.isdigit()


def test_every_option_reaches_the_command_line(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    cmd = run_jaxqtl(
        dd,
        mode="cis",
        prefix="jx",
        out="res",
        model="poisson",
        test="wald",
        robust_se=True,
        nperm=50,
        window=100_000,
        tss_centered=True,
        covar_name=["sex", "age"],
        normalize_covar=False,
        one_hot=True,
        no_intercept=True,
        expression_pcs=3,
        pcs_normalization="tmm",
        pcs_transform="none",
        pcs_libsize="libsize.tsv",
        offset_name_from_covar="age",
        keep=["D0", "D1"],
        min_indiv_expr_pct=0.2,
        min_gene_expr_pct=0.1,
        chrom="1",
        genes="G1",
        maf=0.05,
        max_iter=50,
        tol=1e-4,
        gtol=1e-5,
        step_size=0.5,
        solver="qr",
        seed=7,
        platform="cpu",
        verbose=True,
        extra_args=["--help"],
        run=False,
    )
    pcs, scan = _commands(cmd)
    pcs_opts = {
        "--num-pcs": ["3"],
        "--normalization": ["tmm"],
        "--transform": ["none"],
        "--libsize": ["libsize.tsv"],
        "--keep": ["jx_keep.txt"],
        "--min-indiv-expr-pct": ["0.2"],
        "--seed": ["7"],
        "--platform": ["cpu"],
        "--out": ["jx_donor_features_pcs.tsv"],
    }
    for flag, values in pcs_opts.items():
        assert _values(pcs, flag) == values, flag
    assert "--verbose" in pcs and "--chr" not in pcs and "--genes" not in pcs

    scan_opts = {
        "--covar": ["jx_donor_features_pcs.tsv"],
        "--model": ["poisson"],
        "--test": ["wald"],
        "--nperm": ["50"],
        "--window": ["100000"],
        "--covar-name": ["sex", "age"],
        "--offset-name-from-covar": ["age"],
        "--keep": ["jx_keep.txt"],
        "--min-indiv-expr-pct": ["0.2"],
        "--min-gene-expr-pct": ["0.1"],
        "--chr": ["1"],
        "--genes": ["G1"],
        "--maf": ["0.05"],
        "--max-iter": ["50"],
        "--tol": ["0.0001"],
        "--gtol": ["1e-05"],
        "--step-size": ["0.5"],
        "--solver": ["qr"],
        "--seed": ["7"],
        "--platform": ["cpu"],
        "--out": ["res"],
    }
    for flag, values in scan_opts.items():
        assert _values(scan, flag) == values, flag
    for flag in ("--robust-se", "--tss-centered", "--one-hot", "--no-intercept", "--verbose"):
        assert flag in scan
    assert "--normalize-covar" not in scan and scan[-1] == "--help"
    assert (tmp_path / "jx_keep.txt").read_text().split() == ["D0", "D1"]


@pytest.mark.parametrize(
    ("mode", "acat", "expected", "absent"),
    [
        ("cis", True, ["--acat"], ["--nperm"]),
        ("nominal", False, [], ["--nperm", "--acat"]),
        ("trans", False, [], ["--nperm", "--acat"]),
    ],
)
def test_modes(tmp_path, monkeypatch, dd, mode, acat, expected, absent):
    monkeypatch.chdir(tmp_path)
    cmd = run_jaxqtl(dd, mode=mode, acat=acat, prefix="jx", expression_pcs=None, run=False, save_cmd_file="job.sh")
    args = shlex.split(cmd)
    assert args[:2] == ["jaxqtl", mode] and "compute-pcs" not in args
    assert all(a in args for a in expected) and not any(a in args for a in absent)
    assert (tmp_path / "job.sh").read_text().strip() == cmd


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "fitnull"}, "mode must be one of"),
        ({"model": "NB"}, "model must be one of"),
        ({"test": "lrt"}, "test must be one of"),
        ({"solver": "lu"}, "solver must be one of"),
        ({"pcs_normalization": "cpm"}, "pcs_normalization must be one of"),
        ({"keep": ["D0"], "exclude": ["D1"]}, "only one of `keep`, `exclude`"),
        ({"covar_name": ["sex"], "rm_covar": ["age"]}, "only one of `covar_name`, `rm_covar`"),
        ({"genes": ["G1"], "exclude_gene_list": "x.txt"}, "only one of `genes`"),
        ({"offset": "o.tsv", "set_offset_from_libsize": True}, "only one of `offset`"),
        ({"pcs_libsize": "l.tsv", "pcs_libsize_name_from_covar": "lib"}, "only one of `pcs_libsize`"),
        ({"robust_se": True}, "requires test='wald'"),
        ({"spa": True, "test": "wald"}, "`spa` applies"),
        ({"spa": True, "model": "gaussian"}, "`spa` applies"),
        ({"acat": True, "mode": "nominal"}, "mode='cis' only"),
        ({"pcs_libsize": "l.tsv", "pcs_normalization": "none"}, "need pcs_normalization"),
    ],
)
def test_invalid_options_raise(tmp_path, monkeypatch, dd, kwargs, match):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=match):
        run_jaxqtl(dd, prefix="jx", run=False, **kwargs)


def test_unknown_options_raise(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_jaxqtl(dd, prefix="jx", run=False, addpc=2)


def test_offsets_and_sample_files(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    args = shlex.split(run_jaxqtl(dd, prefix="jx", offset="o.tsv", exclude="drop.txt", expression_pcs=None, run=False))
    assert _values(args, "--offset") == ["o.tsv"] and _values(args, "--exclude") == ["drop.txt"]
    assert "--set-offset-from-libsize" not in args and "--keep" not in args
    args = shlex.split(run_jaxqtl(dd, prefix="jx", set_offset_from_libsize=True, expression_pcs=None, run=False))
    assert "--set-offset-from-libsize" in args and "--offset" not in args


def test_text_covariates_need_one_hot(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    dd.G.obs["batch"] = np.resize(["a", "b", "c"], dd.G.n_obs)
    with pytest.raises(ValueError, match="one_hot=True"):
        run_jaxqtl(dd, prefix="jx", additional_covariates=["batch"], expression_pcs=None, run=False)
    run_jaxqtl(dd, prefix="jx", additional_covariates=["batch"], one_hot=True, expression_pcs=None, run=False)
    cov = pd.read_csv("jx_donor_features.tsv", sep="\t", index_col="iid")
    assert cov.loc[dd.G.obs_names, "batch"].tolist() == dd.G.obs["batch"].tolist()


def _write(df, path):
    df.to_pickle(path)  # pickled_parquet reads pickles at .parquet paths


def test_read_jaxqtl_results(tmp_path, pickled_parquet):
    cis = pd.DataFrame({"phenotype_id": ["g1"], "snp": ["1_10_A_G"], "pvalue": [0.01], "pvalue_adj": [0.04]})
    _write(cis, tmp_path / "jx.cis.score.perm.parquet.gz")
    acat = cis.assign(pvalue_adj=0.02)
    _write(acat, tmp_path / "jx.cis.score.acat.parquet.gz")
    spa = cis.assign(pvalue=0.03)
    _write(spa, tmp_path / "jx.cis.score.spa.perm.parquet.gz")
    nominal = pd.DataFrame({"phenotype_id": ["g1", "g1"], "snp": ["1_10_A_G", "1_20_C_T"], "pvalue": [0.01, 0.5]})
    _write(nominal, tmp_path / "jx.nominal.wald.parquet.gz")
    stats = pd.DataFrame({"phenotype": ["g1", "g2"], "snp": ["1_10_A_G", "1_10_A_G"], "pvalue": [0.2, 0.3]})
    _write(stats, tmp_path / "jx.trans.score.sumstats.parquet.gz")
    info = pd.DataFrame({"chrom": ["1"], "snp": ["1_10_A_G"], "pos": [10], "a1": ["A"], "a0": ["G"]})
    _write(info, tmp_path / "jx.trans.score.variant.info.parquet.gz")

    prefix = str(tmp_path / "jx")
    pd.testing.assert_frame_equal(read_jaxqtl_results(prefix), cis)
    pd.testing.assert_frame_equal(read_jaxqtl_results(prefix, acat=True), acat)
    pd.testing.assert_frame_equal(read_jaxqtl_results(prefix, test="score.spa"), spa)
    pd.testing.assert_frame_equal(read_jaxqtl_results(prefix, mode="nominal", test="wald"), nominal)
    trans = read_jaxqtl_results(prefix, mode="trans")
    assert list(trans.columns) == ["phenotype", "snp", "pvalue", "chrom", "pos", "a1", "a0"]
    assert trans["pos"].tolist() == [10, 10]
    with pytest.raises(FileNotFoundError, match="no gene passes"):
        read_jaxqtl_results(prefix, mode="nominal")


@pytest.mark.parametrize(
    ("kwargs", "result"),
    [
        ({}, "res.cis.score.perm.parquet.gz"),
        ({"acat": True}, "res.cis.score.acat.parquet.gz"),
        ({"spa": True}, "res.cis.score.spa.perm.parquet.gz"),
        ({"mode": "nominal", "test": "wald"}, "res.nominal.wald.parquet.gz"),
    ],
)
def test_run_reads_the_files_jaxqtl_writes(tmp_path, monkeypatch, dd, pickled_parquet, kwargs, result):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jx.shutil, "which", lambda name: f"/usr/bin/{name}")
    commands = []

    def fake_jaxqtl(cmd, check=False, **_):
        commands.append(cmd)
        out = cmd[cmd.index("--out") + 1]
        if cmd[1] == "compute-pcs":
            covar = pd.read_csv(cmd[cmd.index("--covar") + 1], sep="\t")
            covar.assign(ExprPC1=0.0).to_csv(out, sep="\t", index=False)
            Path(f"{out}.variance.tsv").write_text("component\tvariance\n")
        else:
            _write(pd.DataFrame({"phenotype_id": ["G0"], "pvalue": [0.01]}), result)
        Path(f"{out}.log").write_text("done\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_jaxqtl)
    Path("donors.txt").write_text("D0\n")
    paths = run_jaxqtl(dd, prefix="jx", out="res", keep=Path("donors.txt"), read_results=False, **kwargs)
    assert paths == [result, "res.log"]
    assert [c[1] for c in commands] == ["compute-pcs", kwargs.get("mode", "cis")]
    # the inputs written for jaxqtl are removed, the user's donor file is kept
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([result, "res.log", "donors.txt"])

    results = run_jaxqtl(dd, prefix="jx", out="res", expression_pcs=None, **kwargs)
    assert results["phenotype_id"].tolist() == ["G0"]
