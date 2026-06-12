"""
Tests for all new cellink.tl.external functions:
  - make_annot_from_bimfile
  - make_annot_from_donor_data  (skipped: needs DonorData env)
  - scores_to_gmt
  - scores_to_covar
  - genesets_dir_to_entrez_gmt
  - run_magma_annotate
  - run_magma_gene_analysis
  - run_magma_gsa
  - run_magma_gpa  (joint and univariate)

Run from the repo root:
    conda run -n singlecell python cellink/test_new_functions.py
"""

import gzip
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Load module files directly via importlib to bypass cellink/__init__.py,
# which requires mudata, zarr, and other heavy deps not installed here.
import importlib.util

_SRC = Path(__file__).parent / "src"

def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, _SRC / rel_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_ldsc2magma = _load("cellink_ldsc2magma", "cellink/tl/external/_ldsc2magma.py")
scores_to_gmt          = _ldsc2magma.scores_to_gmt
scores_to_covar        = _ldsc2magma.scores_to_covar
genesets_dir_to_entrez_gmt = _ldsc2magma.genesets_dir_to_entrez_gmt
run_magma_annotate     = _ldsc2magma.run_magma_annotate
run_magma_gene_analysis = _ldsc2magma.run_magma_gene_analysis
run_magma_gsa          = _ldsc2magma.run_magma_gsa
run_magma_gpa          = _ldsc2magma.run_magma_gpa

# _ldsc.py imports DonorData, cellink.io.to_plink, cellink.resources._utils
# Stub them all out so the module loads without the full cellink install.
import types

def _stub_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

class _DonorData: pass

_stub_module("cellink._core", DonorData=_DonorData)
_stub_module("cellink",
             _core=sys.modules["cellink._core"])
_stub_module("cellink.io", to_plink=None)
_stub_module("cellink.resources", _utils=None)
_stub_module("cellink.resources._utils", get_data_home=lambda *a, **kw: "/tmp")

_ldsc = _load("cellink_ldsc", "cellink/tl/external/_ldsc.py")
make_annot_from_bimfile = _ldsc.make_annot_from_bimfile

# ---------------------------------------------------------------------------
# Known real paths
# ---------------------------------------------------------------------------
GENE_LOC = "/project/genomics/ayshan/ldsc_analysis/data_2/annotation_sources/Genecode/gencode_v39_grch38_ensg.gene.loc"
BIM22    = "/project/genomics/ayshan/ldsc_analysis/data_2/pre_annotations/annot_generation/1000G_EUR_Phase3_plink/1000G.EUR.QC.22.bim"
GENES_RAW = "/project/genomics/ayshan/ldsc_analysis/MODEL/INPUT/magma_zscore/scz_result.genes.raw"
MAGMA_BIN = "/project/genomics/ayshan/ldsc_analysis/psychmagma/magma"
SCORES_CSV = "/project/genomics/ayshan/ldsc_analysis/data_2/single_cell_pre_annotations/1k1k/seismic/scores.csv"

PASS = []
FAIL = []


def ok(name):
    print(f"  [PASS] {name}")
    PASS.append(name)


def fail(name, exc):
    print(f"  [FAIL] {name}: {exc}")
    FAIL.append(name)


# ---------------------------------------------------------------------------
# Helpers: build minimal synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_scores(n_genes=200, n_cts=5, seed=0):
    """Return a genes × cell-types DataFrame with ENSG-style row IDs."""
    rng = np.random.default_rng(seed)
    genes = [f"ENSG{i:011d}" for i in range(1, n_genes + 1)]
    cts   = [f"CT{i}" for i in range(1, n_cts + 1)]
    data  = rng.random((n_genes, n_cts))
    data[rng.integers(0, n_genes, 20), :] = np.nan  # sprinkle some NaN
    return pd.DataFrame(data, index=genes, columns=cts)


def make_synthetic_bim(chrom=22, n_snps=300, seed=1):
    """Return a DataFrame in PLINK .bim format."""
    rng  = np.random.default_rng(seed)
    bps  = np.sort(rng.integers(16_000_000, 51_000_000, n_snps))
    rows = {
        "CHR": chrom,
        "SNP": [f"rs{i}" for i in range(n_snps)],
        "CM":  np.zeros(n_snps),
        "BP":  bps,
        "A1":  "A",
        "A2":  "G",
    }
    return pd.DataFrame(rows)[["CHR", "SNP", "CM", "BP", "A1", "A2"]]


def make_synthetic_gene_coord(chrom=22, n_genes=50, seed=2):
    """Return a headless gene coordinate DataFrame with ENSG IDs."""
    rng    = np.random.default_rng(seed)
    starts = np.sort(rng.integers(16_000_000, 50_000_000, n_genes))
    ends   = starts + rng.integers(1_000, 100_000, n_genes)
    genes  = [f"ENSG{i:011d}" for i in range(1, n_genes + 1)]
    return pd.DataFrame({
        "gene":  genes,
        "chr":   str(chrom),
        "start": starts,
        "end":   ends,
    })


# ---------------------------------------------------------------------------
# Test 1: make_annot_from_bimfile (synthetic data)
# ---------------------------------------------------------------------------

def test_continuous_annot_synthetic():

    scores_df = make_synthetic_scores(n_genes=50)
    bim_df    = make_synthetic_bim()
    coord_df  = make_synthetic_gene_coord()

    with tempfile.TemporaryDirectory() as tmp:
        bim_path   = os.path.join(tmp, "test.22.bim")
        coord_path = os.path.join(tmp, "gene_coord.txt")
        annot_path = os.path.join(tmp, "CT1.22.annot.gz")

        bim_df.to_csv(bim_path, sep="\t", index=False, header=False)
        coord_df.to_csv(coord_path, sep="\t", index=False, header=False)

        # use first cell type column as a Series
        scores = scores_df["CT1"].dropna()

        result = make_annot_from_bimfile(
            bimfile=bim_path,
            scores=scores,
            annot_file=annot_path,
            gene_coord_file=coord_path,
            windowsize=100_000,
            score_agg="max",
        )

        assert os.path.exists(annot_path), "annot file not created"
        out = pd.read_csv(annot_path, sep="\t")
        assert list(out.columns) == ["CHR", "BP", "SNP", "CM", "ANNOT"], f"bad columns: {list(out.columns)}"
        assert len(out) == len(bim_df), "row count mismatch"
        assert (out["ANNOT"] >= 0).all(), "negative ANNOT values"
        assert result["n_nonzero_snps"] >= 0
        assert result["n_genes_matched"] >= 0

    ok("make_annot_from_bimfile (synthetic)")


def test_continuous_annot_score_agg_variants():
    """max / sum / mean should all produce valid output with same shape."""

    scores_df = make_synthetic_scores(n_genes=50)
    bim_df    = make_synthetic_bim()
    coord_df  = make_synthetic_gene_coord()

    with tempfile.TemporaryDirectory() as tmp:
        bim_path   = os.path.join(tmp, "test.22.bim")
        coord_path = os.path.join(tmp, "gc.txt")
        bim_df.to_csv(bim_path, sep="\t", index=False, header=False)
        coord_df.to_csv(coord_path, sep="\t", index=False, header=False)
        scores = scores_df["CT1"].dropna()

        results = {}
        for agg in ("max", "sum", "mean"):
            out_path = os.path.join(tmp, f"CT1_{agg}.annot.gz")
            res = make_annot_from_bimfile(
                bimfile=bim_path, scores=scores, annot_file=out_path,
                gene_coord_file=coord_path, score_agg=agg,
            )
            df = pd.read_csv(out_path, sep="\t")
            results[agg] = df["ANNOT"].values

        assert len(results["max"]) == len(bim_df)
        # sum >= max for genes with overlapping windows
        assert (results["sum"] >= results["max"] - 1e-12).all(), "sum should be >= max"

    ok("make_annot_from_bimfile score_agg variants")


def test_continuous_annot_real_bim():
    """Run against the real chr22 1000G bimfile and gene.loc."""

    if not os.path.exists(BIM22) or not os.path.exists(GENE_LOC):
        print("  [SKIP] make_continuous_annot real bim: files missing")
        return

    # Load a handful of real ENSG scores from the seismic file
    scores_full = pd.read_csv(SCORES_CSV, index_col=0)
    col = scores_full.columns[0]
    scores = scores_full[col].dropna().head(2000)

    with tempfile.TemporaryDirectory() as tmp:
        annot_path = os.path.join(tmp, "real_ct.22.annot.gz")
        result = make_annot_from_bimfile(
            bimfile=BIM22,
            scores=scores,
            annot_file=annot_path,
            gene_coord_file=GENE_LOC,
            windowsize=100_000,
        )
        out = pd.read_csv(annot_path, sep="\t")
        assert list(out.columns) == ["CHR", "BP", "SNP", "CM", "ANNOT"]
        assert (out["ANNOT"] >= 0).all()
        print(f"    real bim chr22: {result['n_nonzero_snps']:,} non-zero SNPs / {len(out):,} total, "
              f"{result['n_genes_matched']:,} genes matched")

    ok("make_annot_from_bimfile (real chr22 bim)")


# ---------------------------------------------------------------------------
# Test 2: scores_to_gmt
# ---------------------------------------------------------------------------

def test_scores_to_gmt_basic():

    scores = make_synthetic_scores(n_genes=200, n_cts=5)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "test.gmt")
        result = scores_to_gmt(scores, out, top_frac=0.10)

        assert result.exists()
        lines = result.read_text().strip().splitlines()
        assert len(lines) == 5, f"expected 5 sets, got {len(lines)}"

        for line in lines:
            parts = line.split("\t")
            assert parts[1] == "NA", "second field should be NA"
            assert len(parts) >= 3, "must have at least one gene"
            # all gene IDs should look like ENSG
            for g in parts[2:]:
                assert g.startswith("ENSG"), f"non-ENSG gene: {g}"

    ok("scores_to_gmt basic")


def test_scores_to_gmt_top_frac():

    scores = make_synthetic_scores(n_genes=100, n_cts=3)

    with tempfile.TemporaryDirectory() as tmp:
        out10  = os.path.join(tmp, "top10.gmt")
        out20  = os.path.join(tmp, "top20.gmt")
        scores_to_gmt(scores, out10, top_frac=0.10)
        scores_to_gmt(scores, out20, top_frac=0.20)

        lines10 = Path(out10).read_text().strip().splitlines()
        lines20 = Path(out20).read_text().strip().splitlines()
        genes10 = len(lines10[0].split("\t")) - 2
        genes20 = len(lines20[0].split("\t")) - 2
        assert genes20 == 2 * genes10, f"expected 20 genes at 20%, got {genes20}"

    ok("scores_to_gmt top_frac scaling")


def test_scores_to_gmt_ascending():

    scores = make_synthetic_scores(n_genes=100, n_cts=2)

    with tempfile.TemporaryDirectory() as tmp:
        out_top = os.path.join(tmp, "top.gmt")
        out_bot = os.path.join(tmp, "bot.gmt")
        scores_to_gmt(scores, out_top, top_frac=0.10, ascending=False)
        scores_to_gmt(scores, out_bot, top_frac=0.10, ascending=True)

        top_genes = set(Path(out_top).read_text().strip().splitlines()[0].split("\t")[2:])
        bot_genes = set(Path(out_bot).read_text().strip().splitlines()[0].split("\t")[2:])
        assert top_genes.isdisjoint(bot_genes), "top and bottom sets should not overlap"

    ok("scores_to_gmt ascending mode")


def test_scores_to_gmt_with_prefix():

    scores = make_synthetic_scores(n_genes=50, n_cts=2)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "prefixed.gmt")
        scores_to_gmt(scores, out, top_frac=0.10, set_name_prefix="myprefix")
        lines = Path(out).read_text().strip().splitlines()
        for line in lines:
            assert line.startswith("myprefix_"), f"prefix not applied: {line[:30]}"

    ok("scores_to_gmt set_name_prefix")


def test_scores_to_gmt_gene_map():
    """symbol → ENSG mapping: only mapped ENSG rows should appear in output."""

    # Use gene symbols as index
    genes   = [f"GENE{i}" for i in range(50)]
    ensg_ids= [f"ENSG{i:011d}" for i in range(1, 51)]
    scores  = pd.DataFrame(
        np.random.default_rng(0).random((50, 2)),
        index=genes, columns=["CT1", "CT2"]
    )
    gene_map = pd.Series(ensg_ids, index=genes)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "mapped.gmt")
        scores_to_gmt(scores, out, top_frac=0.10, gene_map=gene_map)
        lines = Path(out).read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            for g in line.split("\t")[2:]:
                assert g.startswith("ENSG"), f"non-ENSG after mapping: {g}"

    ok("scores_to_gmt gene_map translation")


# ---------------------------------------------------------------------------
# Test 3: scores_to_covar
# ---------------------------------------------------------------------------

def test_scores_to_covar_basic():

    scores = make_synthetic_scores(n_genes=100, n_cts=4)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "test.covar")
        result = scores_to_covar(scores, out)

        assert result.exists()
        df = pd.read_csv(out, sep="\t", index_col=0)
        assert df.index.name == "GENE"
        assert df.shape[1] == 4
        assert (df.index.str.startswith("ENSG")).all(), "non-ENSG rows in covar"

    ok("scores_to_covar basic")


def test_scores_to_covar_negate():

    scores = make_synthetic_scores(n_genes=50, n_cts=2)

    with tempfile.TemporaryDirectory() as tmp:
        out_pos = os.path.join(tmp, "pos.covar")
        out_neg = os.path.join(tmp, "neg.covar")
        scores_to_covar(scores, out_pos, negate=False)
        scores_to_covar(scores, out_neg, negate=True)

        df_pos = pd.read_csv(out_pos, sep="\t", index_col=0).astype(float)
        df_neg = pd.read_csv(out_neg, sep="\t", index_col=0).astype(float)
        assert np.allclose(df_pos.values, -df_neg.values, equal_nan=True), \
            "negated covar should be sign-flipped"

    ok("scores_to_covar negate")


def test_scores_to_covar_dedup():
    """Duplicate ENSG IDs should be deduplicated (keep highest mean |score|)."""

    genes  = ["ENSG00000000001"] * 3 + [f"ENSG{i:011d}" for i in range(2, 20)]
    scores = pd.DataFrame(
        np.random.default_rng(7).random((len(genes), 2)),
        index=genes, columns=["CT1", "CT2"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "dedup.covar")
        scores_to_covar(scores, out)
        df = pd.read_csv(out, sep="\t", index_col=0)
        assert not df.index.duplicated().any(), "duplicates not removed"
        assert "ENSG00000000001" in df.index

    ok("scores_to_covar dedup")


def test_scores_to_covar_gene_map():
    """Symbol-indexed scores should be remapped to ENSG."""

    genes    = [f"GENE{i}" for i in range(30)]
    ensg_ids = [f"ENSG{i:011d}" for i in range(1, 31)]
    scores   = pd.DataFrame(
        np.random.default_rng(3).random((30, 2)),
        index=genes, columns=["CT1", "CT2"],
    )
    gene_map = pd.Series(ensg_ids, index=genes)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "mapped.covar")
        scores_to_covar(scores, out, gene_map=gene_map)
        df = pd.read_csv(out, sep="\t", index_col=0)
        assert (df.index.str.startswith("ENSG")).all()

    ok("scores_to_covar gene_map translation")


# ---------------------------------------------------------------------------
# Test 4: genesets_dir_to_entrez_gmt
# ---------------------------------------------------------------------------

def make_geneset_dir(tmp, n_sets=3, n_genes=20):
    gdir = os.path.join(tmp, "genesets")
    os.makedirs(gdir, exist_ok=True)
    gene_pool = [f"ENSG{i:011d}" for i in range(1, 200)]
    rng = np.random.default_rng(9)
    for i in range(n_sets):
        fname = os.path.join(gdir, f"CellType{i}.GeneSet")
        chosen = rng.choice(gene_pool, n_genes, replace=False)
        Path(fname).write_text("\n".join(chosen) + "\n")
    return gdir


def test_genesets_dir_to_entrez_gmt_basic():

    with tempfile.TemporaryDirectory() as tmp:
        gdir = make_geneset_dir(tmp, n_sets=3, n_genes=20)
        out  = os.path.join(tmp, "output.gmt")
        result = genesets_dir_to_entrez_gmt(geneset_dir=gdir, out_gmt=out)

        assert result.exists()
        lines = result.read_text().strip().splitlines()
        assert len(lines) == 3, f"expected 3 sets, got {len(lines)}"
        for line in lines:
            parts = line.split("\t")
            assert parts[1] == "S-LDSC derived gene set"
            assert len(parts) >= 4, "too few genes"

    ok("genesets_dir_to_entrez_gmt basic")


def test_genesets_dir_to_entrez_gmt_default_output():
    """When out_gmt is None, should write to sibling magma_genesets/genesets.gmt."""

    with tempfile.TemporaryDirectory() as tmp:
        gdir   = make_geneset_dir(tmp, n_sets=2, n_genes=10)
        result = genesets_dir_to_entrez_gmt(geneset_dir=gdir, out_gmt=None)

        # genesets_dir_to_entrez_gmt calls .resolve() on geneset_dir, so match that
        expected = Path(gdir).resolve().parent / "magma_genesets" / "genesets.gmt"
        assert result == expected, f"default path wrong: {result}"
        assert result.exists()

    ok("genesets_dir_to_entrez_gmt default output path")


def test_genesets_dir_to_entrez_gmt_exclude_control():

    with tempfile.TemporaryDirectory() as tmp:
        gdir = make_geneset_dir(tmp, n_sets=3, n_genes=10)
        # Add a Control.GeneSet
        Path(os.path.join(gdir, "Control.GeneSet")).write_text(
            "\n".join([f"ENSG{i:011d}" for i in range(300, 320)]) + "\n"
        )
        out_excl = os.path.join(tmp, "no_ctrl.gmt")
        out_incl = os.path.join(tmp, "with_ctrl.gmt")
        genesets_dir_to_entrez_gmt(geneset_dir=gdir, out_gmt=out_excl, include_control=False)
        genesets_dir_to_entrez_gmt(geneset_dir=gdir, out_gmt=out_incl, include_control=True)

        n_excl = len(Path(out_excl).read_text().strip().splitlines())
        n_incl = len(Path(out_incl).read_text().strip().splitlines())
        assert n_incl == n_excl + 1, f"control set not added: {n_excl} vs {n_incl}"

    ok("genesets_dir_to_entrez_gmt include_control")


def test_genesets_dir_to_entrez_gmt_real_genesets():
    """Run against GTEx brain GeneSet files that exist on disk."""

    gdir = "/project/genomics/ayshan/ldsc_analysis/data_2/annotations/GTEx_brain_1000Gv3_ldscores"
    if not os.path.isdir(gdir):
        print("  [SKIP] genesets_dir_to_entrez_gmt real: dir missing")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "gtex.gmt")
        result = genesets_dir_to_entrez_gmt(geneset_dir=gdir, out_gmt=out)
        lines = result.read_text().strip().splitlines()
        print(f"    GTEx brain: {len(lines)} gene sets written")
        assert len(lines) > 0

    ok("genesets_dir_to_entrez_gmt real GTEx GeneSet files")


# ---------------------------------------------------------------------------
# Test 5: run_magma_annotate (run=False — command inspection)
# ---------------------------------------------------------------------------

def test_run_magma_annotate_dry_run():

    result = run_magma_annotate(
        snp_loc="gwas_snps.txt",
        gene_loc="NCBI38.gene.loc",
        out_prefix="results/test",
        magma_bin=MAGMA_BIN,
        window_kb=35,
        run=False,
    )
    cmd = result["command"]
    assert cmd[0] == MAGMA_BIN
    assert "--annotate" in cmd
    assert "window=35" in cmd
    assert "--snp-loc" in cmd
    assert "--gene-loc" in cmd
    assert "--out" in cmd

    ok("run_magma_annotate dry-run command")


def test_run_magma_annotate_no_window():

    result = run_magma_annotate(
        snp_loc="snps.txt", gene_loc="genes.loc", out_prefix="out",
        magma_bin=MAGMA_BIN, window_kb=0, run=False,
    )
    assert "window=0" not in result["command"], "window=0 should not appear in command"

    ok("run_magma_annotate window_kb=0 omitted")


# ---------------------------------------------------------------------------
# Test 6: run_magma_gene_analysis (run=False)
# ---------------------------------------------------------------------------

def test_run_magma_gene_analysis_dry_run():

    result = run_magma_gene_analysis(
        bfile="g1000_eur/g1000_eur",
        pval_file="scz.txt",
        gene_annot="scz.genes.annot",
        out_prefix="results/scz",
        n_samples=67_390,
        magma_bin=MAGMA_BIN,
        run=False,
    )
    cmd = result["command"]
    assert "--bfile" in cmd
    assert "--pval" in cmd
    assert "--gene-annot" in cmd
    assert any("N=67390" in c for c in cmd), f"N= not in cmd: {cmd}"

    ok("run_magma_gene_analysis dry-run command")


def test_run_magma_gene_analysis_no_n():
    """When n_samples is None, N= should not appear in the pval argument."""

    result = run_magma_gene_analysis(
        bfile="g1000_eur", pval_file="scz.txt", gene_annot="scz.annot",
        out_prefix="out", n_samples=None, magma_bin=MAGMA_BIN, run=False,
    )
    cmd = result["command"]
    assert not any("N=" in c for c in cmd), "N= should not appear when n_samples=None"

    ok("run_magma_gene_analysis n_samples=None")


# ---------------------------------------------------------------------------
# Test 7: run_magma_gsa (run=False + real execution if genes.raw available)
# ---------------------------------------------------------------------------

def test_run_magma_gsa_dry_run():

    result = run_magma_gsa(
        gene_results="scz.genes.raw",
        set_annot="genesets.gmt",
        out_prefix="results/gsa",
        magma_bin=MAGMA_BIN,
        run=False,
    )
    cmd = result["command"]
    assert "--gene-results" in cmd
    assert "--set-annot" in cmd
    assert "--out" in cmd

    ok("run_magma_gsa dry-run command")


def test_run_magma_gsa_real():
    """Execute real GSA using existing genes.raw and a synthetic GMT."""

    if not os.path.exists(GENES_RAW) or not os.path.exists(MAGMA_BIN):
        print("  [SKIP] run_magma_gsa real: files missing")
        return

    # Build a minimal GMT from genes actually in genes.raw
    gene_ids = []
    with open(GENES_RAW) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            gene_ids.append(parts[0])
            if len(gene_ids) >= 500:
                break

    rng = np.random.default_rng(42)
    set1 = rng.choice(gene_ids, 100, replace=False).tolist()
    set2 = rng.choice(gene_ids, 100, replace=False).tolist()

    with tempfile.TemporaryDirectory() as tmp:
        gmt_path = os.path.join(tmp, "test.gmt")
        with open(gmt_path, "w") as f:
            f.write("TestSet1\tNA\t" + "\t".join(set1) + "\n")
            f.write("TestSet2\tNA\t" + "\t".join(set2) + "\n")

        result = run_magma_gsa(
            gene_results=GENES_RAW,
            set_annot=gmt_path,
            out_prefix=os.path.join(tmp, "gsa_test"),
            magma_bin=MAGMA_BIN,
        )
        assert os.path.exists(result["results_file"]), "GSA output missing"
        df = pd.read_csv(result["results_file"], sep=r"\s+", comment="#")
        assert "P" in df.columns, f"P column missing from GSA output: {df.columns.tolist()}"
        assert len(df) >= 1
        print(f"    GSA output: {len(df)} rows, P values: {df['P'].values}")

    ok("run_magma_gsa real execution")


# ---------------------------------------------------------------------------
# Test 8: run_magma_gpa joint (run=False + real execution)
# ---------------------------------------------------------------------------

def test_run_magma_gpa_dry_run():

    result = run_magma_gpa(
        gene_results="scz.genes.raw",
        gene_covar="scores.covar",
        out_prefix="results/gpa",
        magma_bin=MAGMA_BIN,
        univariate=False,
        run=False,
    )
    cmd = result["command"]
    assert "--gene-results" in cmd
    assert "--gene-covar" in cmd
    assert "--out" in cmd

    ok("run_magma_gpa dry-run command (joint)")


def test_run_magma_gpa_real_joint():
    """Joint GPA using the real seismic covar file + genes.raw."""

    if not os.path.exists(GENES_RAW) or not os.path.exists(MAGMA_BIN) or not os.path.exists(SCORES_CSV):
        print("  [SKIP] run_magma_gpa real joint: files missing")
        return

    scores_full = pd.read_csv(SCORES_CSV, index_col=0)
    # Use first 5 cell types to keep it fast
    scores = scores_full.iloc[:, :5]

    with tempfile.TemporaryDirectory() as tmp:
        covar_path = os.path.join(tmp, "test.covar")
        scores_to_covar(scores, covar_path)

        result = run_magma_gpa(
            gene_results=GENES_RAW,
            gene_covar=covar_path,
            out_prefix=os.path.join(tmp, "gpa_joint"),
            magma_bin=MAGMA_BIN,
            univariate=False,
        )
        assert os.path.exists(result["results_file"])
        df = pd.read_csv(result["results_file"], sep=r"\s+", comment="#")
        assert "P" in df.columns
        print(f"    GPA joint: {len(df)} cell types tested, top P={df['P'].min():.3e}")

    ok("run_magma_gpa real joint execution")


# ---------------------------------------------------------------------------
# Test 9: run_magma_gpa univariate
# ---------------------------------------------------------------------------

def test_run_magma_gpa_real_univariate():
    """Univariate GPA: each cell type tested independently."""

    if not os.path.exists(GENES_RAW) or not os.path.exists(MAGMA_BIN) or not os.path.exists(SCORES_CSV):
        print("  [SKIP] run_magma_gpa real univariate: files missing")
        return

    scores_full = pd.read_csv(SCORES_CSV, index_col=0)
    scores = scores_full.iloc[:, :3]  # 3 cell types

    with tempfile.TemporaryDirectory() as tmp:
        covar_path = os.path.join(tmp, "test.covar")
        scores_to_covar(scores, covar_path)

        result = run_magma_gpa(
            gene_results=GENES_RAW,
            gene_covar=covar_path,
            out_prefix=os.path.join(tmp, "gpa_univ"),
            magma_bin=MAGMA_BIN,
            univariate=True,
        )
        assert os.path.exists(result["results_file"])
        content = Path(result["results_file"]).read_text()
        assert "UNIVARIATE" in content
        # should have one line per cell type (+ header lines)
        data_lines = [l for l in content.splitlines() if l and not l.startswith("#")]
        print(f"    GPA univariate: {len(data_lines)} cell types in output")
        assert len(data_lines) == 3, f"expected 3 CT lines, got {len(data_lines)}"

    ok("run_magma_gpa real univariate execution")


# ---------------------------------------------------------------------------
# Test 10: end-to-end scores_to_gmt → run_magma_gsa
# ---------------------------------------------------------------------------

def test_e2e_scores_to_gmt_then_gsa():

    if not os.path.exists(GENES_RAW) or not os.path.exists(MAGMA_BIN) or not os.path.exists(SCORES_CSV):
        print("  [SKIP] e2e scores_to_gmt→GSA: files missing")
        return

    scores_full = pd.read_csv(SCORES_CSV, index_col=0)
    scores = scores_full.iloc[:, :4]

    with tempfile.TemporaryDirectory() as tmp:
        gmt_path = os.path.join(tmp, "from_scores.gmt")
        scores_to_gmt(scores, gmt_path, top_frac=0.10)

        result = run_magma_gsa(
            gene_results=GENES_RAW,
            set_annot=gmt_path,
            out_prefix=os.path.join(tmp, "e2e_gsa"),
            magma_bin=MAGMA_BIN,
        )
        assert os.path.exists(result["results_file"])
        df = pd.read_csv(result["results_file"], sep=r"\s+", comment="#")
        assert "P" in df.columns
        print(f"    E2E GSA: {len(df)} sets, top P={df['P'].min():.3e}")

    ok("e2e scores_to_gmt → run_magma_gsa")


# ---------------------------------------------------------------------------
# Test 11: end-to-end scores_to_covar → run_magma_gpa
# ---------------------------------------------------------------------------

def test_e2e_scores_to_covar_then_gpa():

    if not os.path.exists(GENES_RAW) or not os.path.exists(MAGMA_BIN) or not os.path.exists(SCORES_CSV):
        print("  [SKIP] e2e scores_to_covar→GPA: files missing")
        return

    scores_full = pd.read_csv(SCORES_CSV, index_col=0)
    scores = scores_full.iloc[:, :4]

    with tempfile.TemporaryDirectory() as tmp:
        covar_path = os.path.join(tmp, "from_scores.covar")
        scores_to_covar(scores, covar_path)

        result = run_magma_gpa(
            gene_results=GENES_RAW,
            gene_covar=covar_path,
            out_prefix=os.path.join(tmp, "e2e_gpa"),
            magma_bin=MAGMA_BIN,
        )
        assert os.path.exists(result["results_file"])
        df = pd.read_csv(result["results_file"], sep=r"\s+", comment="#")
        print(f"    E2E GPA: {len(df)} cell types, top P={df['P'].min():.3e}")

    ok("e2e scores_to_covar → run_magma_gpa")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_continuous_annot_synthetic,
    test_continuous_annot_score_agg_variants,
    test_continuous_annot_real_bim,
    test_scores_to_gmt_basic,
    test_scores_to_gmt_top_frac,
    test_scores_to_gmt_ascending,
    test_scores_to_gmt_with_prefix,
    test_scores_to_gmt_gene_map,
    test_scores_to_covar_basic,
    test_scores_to_covar_negate,
    test_scores_to_covar_dedup,
    test_scores_to_covar_gene_map,
    test_genesets_dir_to_entrez_gmt_basic,
    test_genesets_dir_to_entrez_gmt_default_output,
    test_genesets_dir_to_entrez_gmt_exclude_control,
    test_genesets_dir_to_entrez_gmt_real_genesets,
    test_run_magma_annotate_dry_run,
    test_run_magma_annotate_no_window,
    test_run_magma_gene_analysis_dry_run,
    test_run_magma_gene_analysis_no_n,
    test_run_magma_gsa_dry_run,
    test_run_magma_gsa_real,
    test_run_magma_gpa_dry_run,
    test_run_magma_gpa_real_joint,
    test_run_magma_gpa_real_univariate,
    test_e2e_scores_to_gmt_then_gsa,
    test_e2e_scores_to_covar_then_gpa,
]

if __name__ == "__main__":
    print(f"\nRunning {len(ALL_TESTS)} tests\n{'='*60}")
    for test_fn in ALL_TESTS:
        print(f"\n→ {test_fn.__name__}")
        try:
            test_fn()
        except Exception as exc:
            fail(test_fn.__name__, exc)
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("Failed tests:")
        for name in FAIL:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("All tests passed.")
