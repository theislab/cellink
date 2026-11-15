import gzip
import shlex

import numpy as np
import pandas as pd
import pytest

from cellink._core.donordata import DonorData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.tl.external._jaxqtl import read_jaxqtl_results, run_jaxqtl


@pytest.fixture
def dd():
    adata = sim_adata(strategy="poisson")  # counts, as jaxQTL's count models expect
    gdata = sim_gdata(adata=adata)
    gdata.obs["donor_id"] = gdata.obs.index
    gdata.obs["sex"] = np.resize([1, 2], gdata.n_obs)
    gdata.obs["age"] = np.linspace(30.5, 70.5, gdata.n_obs)
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


def _phenotype(prefix):
    df = pd.read_csv(f"{prefix}_phenotype.bed.gz", sep="\t", index_col=3)
    return df.drop(columns=["#chr", "start", "end"]).T


def _per_donor(dd, func, values=None):
    values = dd.C.X if values is None else values
    return (
        pd.DataFrame(values, index=dd.C.obs_names, columns=dd.C.var_names)
        .groupby(dd.C.obs["donor_id"].to_numpy())
        .agg(func)
    )


@pytest.mark.parametrize("model", ["nb", "poisson"])
def test_count_models_get_summed_counts_and_libsize_offset(tmp_path, monkeypatch, dd, model, caplog):
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("WARNING", logger="cellink.tl.external._jaxqtl"):
        cmd = run_jaxqtl(dd, prefix="jx", model=model, expression_pcs=None, run=False)
    assert "--set-offset-from-libsize" in shlex.split(cmd)
    expected = _per_donor(dd, "sum").loc[dd.G.obs_names]
    np.testing.assert_allclose(_phenotype("jx").loc[dd.G.obs_names, dd.C.var_names], expected)
    assert not any("expects raw counts" in m for m in caplog.messages)


def test_gaussian_keeps_mean_without_offset(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    cmd = run_jaxqtl(dd, prefix="jx", model="gaussian", expression_pcs=None, run=False)
    assert "--set-offset-from-libsize" not in shlex.split(cmd)
    expected = _per_donor(dd, "mean").loc[dd.G.obs_names]
    np.testing.assert_allclose(_phenotype("jx").loc[dd.G.obs_names, dd.C.var_names], expected, rtol=1e-5)


def test_offset_options(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    args = shlex.split(run_jaxqtl(dd, prefix="jx", offset="offsets.tsv", expression_pcs=None, run=False))
    assert args[args.index("--offset") + 1] == "offsets.tsv" and "--set-offset-from-libsize" not in args
    args = shlex.split(run_jaxqtl(dd, prefix="jx", set_offset_from_libsize=False, expression_pcs=None, run=False))
    assert "--set-offset-from-libsize" not in args
    dd.G.obs["log_libsize"] = np.log(_per_donor(dd, "sum").sum(axis=1).loc[dd.G.obs_names].to_numpy())
    kwargs = {"additional_covariates": ["log_libsize"], "offset_name_from_covar": "log_libsize"}
    args = shlex.split(run_jaxqtl(dd, prefix="jx", expression_pcs=None, run=False, **kwargs))
    assert args[args.index("--offset-name-from-covar") + 1] == "log_libsize"
    assert "--set-offset-from-libsize" not in args
    with pytest.raises(ValueError, match="only one of `offset`"):
        run_jaxqtl(dd, prefix="jx", offset="offsets.tsv", set_offset_from_libsize=True, run=False)


def test_counts_layer_and_warning_on_normalized_data(tmp_path, monkeypatch, dd, caplog):
    monkeypatch.chdir(tmp_path)
    counts = dd.C.X.copy()
    dd.C.layers["counts"] = counts
    dd.C.X = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)

    with caplog.at_level("WARNING", logger="cellink.tl.external._jaxqtl"):
        run_jaxqtl(dd, prefix="jx", counts_layer="counts", expression_pcs=None, run=False)
    assert not any("expects raw counts" in m for m in caplog.messages)
    expected = _per_donor(dd, "sum", counts).loc[dd.G.obs_names]
    np.testing.assert_allclose(_phenotype("jx").loc[dd.G.obs_names, dd.C.var_names], expected)

    with caplog.at_level("WARNING", logger="cellink.tl.external._jaxqtl"):
        run_jaxqtl(dd, prefix="jx", expression_pcs=None, run=False)
    assert any("expects raw counts" in m and "dd.C.X" in m for m in caplog.messages)


def test_covariates_keep_float_age_and_reject_missing_sex(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    run_jaxqtl(dd, prefix="jx", expression_pcs=None, run=False)
    cov = pd.read_csv("jx_donor_features.tsv", sep="\t", index_col="iid")
    np.testing.assert_allclose(cov["age"], dd.G.obs["age"], rtol=1e-6)
    assert set(cov["sex"]) == {0, 1}

    run_jaxqtl(dd, prefix="jx", encode_sex=False, encode_age=False, expression_pcs=None, run=False)
    assert pd.read_csv("jx_donor_features.tsv", sep="\t").columns.tolist() == ["iid"]

    dd.G.obs["sex"] = dd.G.obs["sex"].astype(float)
    dd.G.obs.iloc[0, dd.G.obs.columns.get_loc("sex")] = np.nan
    with pytest.raises(ValueError, match="have no sex"):
        run_jaxqtl(dd, prefix="jx", expression_pcs=None, run=False)


def test_libsize_key_gives_offsets_and_pca_library_sizes_from_all_genes(tmp_path, monkeypatch, dd):
    monkeypatch.chdir(tmp_path)
    # the cells' totals over all genes, not only the genes in dd.C
    dd.C.obs["nCount_RNA"] = np.asarray(dd.C.X.sum(axis=1)).ravel() * 3.0
    pcs, scan = _commands(run_jaxqtl(dd, prefix="jx", libsize_key="nCount_RNA", run=False))
    expected = dd.C.obs.groupby("donor_id", observed=True)["nCount_RNA"].sum()
    libsize = pd.read_csv("jx_libsize.tsv", sep="\t", index_col="iid")["libsize"]
    np.testing.assert_allclose(libsize.loc[expected.index], expected)
    offset = pd.read_csv("jx_offset.tsv", sep="\t", index_col="iid")["offset"]
    np.testing.assert_allclose(offset.loc[expected.index], np.log(expected))
    assert _values(scan, "--offset") == ["jx_offset.tsv"] and "--set-offset-from-libsize" not in scan
    assert _values(pcs, "--libsize") == ["jx_libsize.tsv"]

    # the Gaussian model has no offset, but the PCA still uses the library sizes
    pcs, scan = _commands(run_jaxqtl(dd, prefix="jx", model="gaussian", libsize_key="nCount_RNA", run=False))
    assert "--offset" not in scan and _values(pcs, "--libsize") == ["jx_libsize.tsv"]

    dd.C.obs.loc[dd.C.obs_names[dd.C.obs["donor_id"] == dd.G.obs_names[0]], "nCount_RNA"] = 0.0
    with pytest.raises(ValueError, match="no positive library size"):
        run_jaxqtl(dd, prefix="jx", libsize_key="nCount_RNA", run=False)
