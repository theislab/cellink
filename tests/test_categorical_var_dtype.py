import numpy as np
import pandas as pd
import pytest

from cellink._core.dummy_data import sim_gdata
from cellink.io._export import write_variants_to_vcf
from cellink.tl._subset_region import subset_genomic_region


def _str_and_categorical_gdata():
    """Two AnnDatas with byte-identical underlying data, differing only in
    whether chrom/a0/a1 are plain string or category dtype -- sim_gdata()
    draws random alleles internally, so calling it twice independently (as
    an earlier version of this fixture did) compares two different random
    genotypes, not the same data under two dtypes.
    """
    gdata_str = sim_gdata(n_donors=20, n_snps=30)
    gdata_str.var["chrom"] = gdata_str.var["chrom"].astype(str)
    gdata_cat = gdata_str.copy()
    gdata_cat.var["chrom"] = gdata_cat.var["chrom"].astype("category")
    gdata_cat.var["a0"] = gdata_cat.var["a0"].astype("category")
    gdata_cat.var["a1"] = gdata_cat.var["a1"].astype("category")
    return gdata_str, gdata_cat


def test_subset_genomic_region_matches_plain_string_dtype():
    gdata_str, gdata_cat = _str_and_categorical_gdata()

    start, end = int(gdata_str.var["pos"].min()), int(gdata_str.var["pos"].max()) + 1
    sub_str = subset_genomic_region(gdata_str, chrom="1", start=start, end=end)
    sub_cat = subset_genomic_region(gdata_cat, chrom="1", start=start, end=end)

    assert sub_str.shape == sub_cat.shape
    assert list(sub_str.var.index) == list(sub_cat.var.index)


def test_np_unique_and_equality_on_categorical_chrom():
    _, gdata = _str_and_categorical_gdata()
    uniq = np.unique(gdata.var["chrom"])
    assert list(uniq) == ["1"]
    mask = gdata.var["chrom"] == "1"
    assert mask.all()
    assert type(gdata.var["chrom"].iloc[0]) is str


def test_write_variants_to_vcf_identical_with_categorical(tmp_path):
    gdata_str, gdata_cat = _str_and_categorical_gdata()

    out_str, out_cat = tmp_path / "str.vcf", tmp_path / "cat.vcf"
    write_variants_to_vcf(gdata_str, out_file=str(out_str))
    write_variants_to_vcf(gdata_cat, out_file=str(out_cat))
    assert out_str.read_text() == out_cat.read_text()


def test_to_plink_roundtrip_identical_with_categorical(tmp_path):
    bed_reader = pytest.importorskip("bed_reader")
    from cellink.io._export import to_plink

    gdata_str, gdata_cat = _str_and_categorical_gdata()
    gdata_str.obs["donor_id"] = gdata_str.obs.index
    gdata_str.obs["sex"] = 0
    gdata_cat.obs["donor_id"] = gdata_cat.obs.index
    gdata_cat.obs["sex"] = 0

    prefix_str, prefix_cat = str(tmp_path / "str"), str(tmp_path / "cat")
    to_plink(gdata_str, output_prefix=prefix_str)
    to_plink(gdata_cat, output_prefix=prefix_cat)

    b_str = bed_reader.open_bed(prefix_str + ".bed")
    b_cat = bed_reader.open_bed(prefix_cat + ".bed")
    np.testing.assert_array_equal(b_str.read(), b_cat.read())
    assert list(b_str.chromosome) == list(b_cat.chromosome)
    assert list(b_str.allele_1) == list(b_cat.allele_1)
    assert list(b_str.allele_2) == list(b_cat.allele_2)


def test_tensorqtl_input_generator_cis_matches_plain_string_dtype():
    """The real integration risk flagged for this change: cellink's
    run_tensorqtl(use_python_api=True) hands variant_df straight to
    tensorqtl's own genotypeio.InputGeneratorCis, which does
    variant_df['chrom'].unique() / .groupby('chrom') / membership checks --
    code cellink does not control. Verify categorical dtype produces
    byte-identical cis-window results there directly, not just in cellink's
    own call sites.
    """
    tensorqtl_genotypeio = pytest.importorskip("tensorqtl.genotypeio")

    rng = np.random.default_rng(0)
    n_var, n_genes, n_samples = 60, 6, 12
    chrom_str = np.array(["1"] * 30 + ["2"] * 30)
    pos = np.concatenate([np.sort(rng.choice(1_000_000, 30, replace=False)),
                           np.sort(rng.choice(1_000_000, 30, replace=False))])
    variant_df_str = pd.DataFrame({"chrom": chrom_str, "pos": pos},
                                   index=[f"var{i}" for i in range(n_var)])
    variant_df_cat = variant_df_str.copy()
    variant_df_cat["chrom"] = variant_df_cat["chrom"].astype("category")

    phenotype_pos_df = pd.DataFrame({
        "chr": np.array(["1"] * 3 + ["2"] * 3),
        "start": np.sort(rng.choice(1_000_000, n_genes, replace=False)),
    }, index=[f"gene{i}" for i in range(n_genes)])
    phenotype_pos_df["end"] = phenotype_pos_df["start"] + 1000

    genotype_df = pd.DataFrame(rng.integers(0, 3, size=(n_var, n_samples)),
                                index=variant_df_str.index, columns=[f"s{i}" for i in range(n_samples)])
    phenotype_df = pd.DataFrame(rng.normal(size=(n_genes, n_samples)),
                                 index=phenotype_pos_df.index, columns=[f"s{i}" for i in range(n_samples)])

    def cis_ranges_for(variant_df):
        gen = tensorqtl_genotypeio.InputGeneratorCis(
            genotype_df, variant_df, phenotype_df, phenotype_pos_df, window=1_000_000
        )
        return dict(gen.cis_ranges), gen.chrs, gen.phenotype_df.index.tolist()

    ranges_str, chrs_str, kept_str = cis_ranges_for(variant_df_str)
    ranges_cat, chrs_cat, kept_cat = cis_ranges_for(variant_df_cat)

    assert chrs_str == chrs_cat
    assert kept_str == kept_cat
    assert set(ranges_str) == set(ranges_cat)
    for k in ranges_str:
        np.testing.assert_array_equal(ranges_str[k], ranges_cat[k])
