import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cellink._core import GAnn
from cellink.resources import get_onek1k
from cellink.resources._utils import get_data_home

GENERATOR_VERSION = "2"
POSITIVE_CONTROL = {"gene": "ENSG00000183172", "snp": "22_42108675_A_G", "celltype": "CD4 Naive"}


def _get_ensembl_gene_annotations():
    """
    Fetch gene annotations from Ensembl via pybiomart.

    Returns
    -------
    pd.DataFrame
        DataFrame with gene annotations including start, end, and chromosome.
    """
    from pybiomart import Server

    print("  Fetching gene annotations from Ensembl...")
    server = Server(host="http://www.ensembl.org")
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
    ensembl_gene_id_start_end_chr = dataset.query(
        attributes=["ensembl_gene_id", "start_position", "end_position", "chromosome_name"]
    )
    ensembl_gene_id_start_end_chr = ensembl_gene_id_start_end_chr.set_index("Gene stable ID")
    ensembl_gene_id_start_end_chr = ensembl_gene_id_start_end_chr.rename(
        columns={
            "Gene start (bp)": GAnn.start,
            "Gene end (bp)": GAnn.end,
            "Chromosome/scaffold name": GAnn.chrom,
        }
    )
    return ensembl_gene_id_start_end_chr


def _get_onek1k_gene_annotations(data_home=None):
    """
    Read the Ensembl 105 (GRCh38) gene table of the OneK1K Zenodo record.

    Returns
    -------
    pd.DataFrame
        Gene annotations (start, end, chromosome, strand) indexed by Ensembl gene id.
    """
    path = get_data_home(data_home) / "onek1k" / "gene_counts_Ensembl_105_phenotype_metadata.tsv.gz"
    print(f"  Reading gene annotations from {path}...")
    meta = pd.read_csv(path, sep="\t", dtype={"chromosome": str})
    meta = meta.drop_duplicates("gene_id").set_index("gene_id")
    ann = pd.DataFrame(
        {
            GAnn.start: meta["gene_start"].astype("int64"),
            GAnn.end: meta["gene_end"].astype("int64"),
            GAnn.chrom: meta["chromosome"].astype(str),
            GAnn.strand: meta["strand"].map({1: "+", -1: "-"}),
        }
    )
    ann.index.name = None
    return ann


def _check_donor_pairing(dd, n_variants: int = 3000, seed: int = 0, min_concordance: float = 0.98):
    """
    Check that each genotype row belongs to the donor whose cells carry its id.
    """
    chrx = np.where(dd.G.var.chrom.astype(str) == "X")[0]
    if len(chrx) == 0:
        print("  [check] no chrX variants in dd.G, skipping the genetic sex check")
        return None
    maf = dd.G.var["maf"].values[chrx] if "maf" in dd.G.var else np.full(len(chrx), 0.5)
    chrx = chrx[maf >= 0.2]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(chrx, min(n_variants, len(chrx)), replace=False))
    X = dd.G[:, idx].X
    X = np.asarray(X.compute() if hasattr(X, "compute") else X)
    het = (X == 1).mean(1)
    genetic_male = pd.Series((het < 0.02).astype(int), index=dd.G.obs_names)
    cell_sex = dd.C.obs.groupby(dd.donor_id, observed=True)["sex"].first().reindex(dd.G.obs_names)
    ok = cell_sex.notna()
    concordance = float((genetic_male[ok] == cell_sex[ok].astype(int)).mean())
    print(
        f"  [check] genetic sex (chrX heterozygosity) vs. annotated sex: {concordance:.3f} concordant "
        f"({ok.sum()} donors, {len(idx)} chrX variants)"
    )
    if concordance < min_concordance:
        raise RuntimeError(
            f"Only {concordance:.1%} of donors have a genotype-inferred sex matching the sex of their cells: "
            "the genotype rows are not paired with the right donors. Check the donor ids of dd.G and dd.C."
        )
    return concordance


def _check_positive_control_eqtl(dd, min_abs_r: float = 0.5):
    """Check that the strongest OneK1K chr22 cis-eQTL (SMDT1) is visible in the pseudobulk data."""
    gene, snp, celltype = POSITIVE_CONTROL["gene"], POSITIVE_CONTROL["snp"], POSITIVE_CONTROL["celltype"]
    if gene not in dd.C.var_names or snp not in dd.G.var_names:
        print(f"  [check] positive control {gene} / {snp} not in the data, skipping")
        return None
    cells = (dd.C.obs["predicted.celltype.l2"] == celltype).values
    C = dd.C[cells]
    counts = np.asarray(C.X.sum(1)).ravel()
    g = np.asarray(C[:, gene].X.todense() if hasattr(C.X, "todense") else C[:, gene].X).ravel()
    expr = pd.Series(np.log1p(g / np.maximum(counts, 1) * 1e4), index=C.obs[dd.donor_id].values)
    pb = expr.groupby(level=0).mean().reindex(dd.G.obs_names)
    x = dd.G[:, snp].X
    x = np.asarray(x.compute() if hasattr(x, "compute") else x, dtype=float).ravel()
    ok = pb.notna().values
    r = float(np.corrcoef(x[ok], pb.values[ok])[0, 1])
    print(f"  [check] positive control eQTL {gene} ~ {snp} in {celltype}: r = {r:.3f} ({ok.sum()} donors)")
    if abs(r) < min_abs_r:
        raise RuntimeError(
            f"The OneK1K positive-control eQTL {gene} ~ {snp} has r = {r:.2f} (expected about -0.9): "
            "genotypes and expression are not paired correctly."
        )
    return r


def _filter_genes_for_qtl_mapping(dd, qtl_window: int = 500_000):
    """
    Filter genes to only those on chr 1-22 and within QTL mapping range.

    Parameters
    ----------
    dd : DonorData
        Input DonorData object
    qtl_window : int
        Window size (in bp) around SNPs to consider for gene filtering

    Returns
    -------
    DonorData
        Filtered DonorData object
    """
    print(f"  Filtering genes for QTL mapping (±{qtl_window/1e6:.1f}Mb window)...")

    valid_chroms = [str(i) for i in range(1, 23)]
    gene_mask = dd.C.var[GAnn.chrom].isin(valid_chroms)
    print(f"    Genes on chr 1-22: {gene_mask.sum()}/{len(dd.C.var)}")

    genes_in_range = set()

    for chrom in valid_chroms:
        snp_mask = dd.G.var.chrom == chrom
        if snp_mask.sum() == 0:
            continue

        snp_positions = dd.G.var.loc[snp_mask, "pos"].values
        min_snp_pos = snp_positions.min() - qtl_window
        max_snp_pos = snp_positions.max() + qtl_window

        gene_chrom_mask = dd.C.var[GAnn.chrom] == chrom
        genes_on_chrom = dd.C.var[gene_chrom_mask]

        genes_in_window = genes_on_chrom[
            (genes_on_chrom[GAnn.start] <= max_snp_pos) & (genes_on_chrom[GAnn.end] >= min_snp_pos)
        ]

        genes_in_range.update(genes_in_window.index)
        print(f"    Chr {chrom}: {len(genes_in_window)} genes in range")

    gene_filter_mask = dd.C.var.index.isin(genes_in_range)
    dd = dd[..., gene_filter_mask].copy()

    print(f"    Total genes retained: {dd.shape[3]}")

    return dd


def _subsample_cells_per_donor(dd, max_cells_per_donor: int, rng: np.random.Generator):
    """Keep at most ``max_cells_per_donor`` randomly chosen cells of every donor (original order kept)."""
    donors = dd.C.obs[dd.donor_id].astype(str).values
    keep = np.zeros(len(donors), dtype=bool)
    for donor in pd.unique(donors):
        idx = np.where(donors == donor)[0]
        if len(idx) > max_cells_per_donor:
            idx = rng.choice(idx, max_cells_per_donor, replace=False)
        keep[idx] = True
    return dd[:, :, keep, :].copy()


def _compress_h5(src: Path, dst: Path, level: int = 4, min_bytes: int = 1 << 16) -> None:
    """Copy an HDF5 file, gzip-compressing every numeric dataset of at least ``min_bytes``."""

    def copy_attrs(a, b):
        for k, v in a.attrs.items():
            b.attrs[k] = v

    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        copy_attrs(fin, fout)

        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                grp = fout.require_group(name)
                copy_attrs(obj, grp)
                return
            compress = obj.shape != () and obj.dtype.kind in "biuf" and obj.nbytes >= min_bytes
            if compress:
                ds = fout.create_dataset(name, data=obj[()], compression="gzip", compression_opts=level, shuffle=True)
            else:
                fin.copy(obj, fout, name=name)
                ds = fout[name]
            copy_attrs(obj, ds)

        fin.visititems(visit)


def generate_dummy_dataset(
    output_path: str = "./dummy_onek1k_dataset.dd.h5",
    n_donors: int = 300,
    max_cells_per_donor: int | None = 420,
    snp_sample_frac: float = 0.001,
    qtl_window: int = 500_000,
    seed: int = 42,
    verify_checksum: bool = False,
    config_path: str = "onek1k",
    data_home: str | None = None,
    gene_annotation: str = "onek1k",
    compression_level: int = 4,
    dd=None,
):
    """
    Generate a dummy OneK1K dataset for tutorial purposes.

    Parameters
    ----------
    output_path : str
        Path where the dummy dataset will be saved (as .dd.h5 or .dd.zarr)
    n_donors : int
        Number of donors to include (randomly sampled)
    max_cells_per_donor : int or None
        Keep at most this many randomly chosen cells per donor; None keeps all cells
    snp_sample_frac : float
        Fraction of SNPs to keep from chromosomes 1-21 (per chromosome)
    qtl_window : int
        Window size (in bp) for filtering genes relevant to QTL mapping
    seed : int
        Random seed for reproducibility (donor, cell and SNP sampling)
    verify_checksum : bool
        Whether to verify checksums when loading original data
    config_path : str, default="onek1k"
        Name of a config shipped with cellink, or path to a OneK1K configuration file
    data_home : str or None
        cellink data directory holding (or receiving) the full OneK1K data
    gene_annotation : {"onek1k", "ensembl"}
        "onek1k": Ensembl 105 gene table from the OneK1K Zenodo record (offline, fixed);
        "ensembl": current Ensembl release through pybiomart
    compression_level : int
        gzip level for the datasets of a .dd.h5 output (0 = uncompressed)
    dd : DonorData, optional
        Full OneK1K data as returned by ``get_onek1k``; loaded with ``get_onek1k`` if not given.
        It is not modified.

    Returns
    -------
    DonorData
        The generated dummy dataset
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Dummy OneK1K Dataset")
    print("=" * 80)

    if dd is None:
        print("Loading full OneK1K dataset...")
        dd = get_onek1k(config_path=config_path, data_home=data_home, verify_checksum=verify_checksum)
    print(f"Original dataset shape: {dd.shape}")

    print("Checking that genotypes and cells are paired by donor...")
    _check_donor_pairing(dd)

    print(f"Sampling {n_donors} random donors...")
    all_donors = np.sort(dd.G.obs.index.values)
    selected_donors = all_donors
    if len(all_donors) > n_donors:
        selected_donors = rng.choice(all_donors, n_donors, replace=False)
    dd = dd[selected_donors, :, :, :].copy()
    print(f"After donor sampling: {dd.shape}")

    print("Adding gene annotations...")
    if gene_annotation == "onek1k":
        gene_ann = _get_onek1k_gene_annotations(data_home)
    elif gene_annotation == "ensembl":
        gene_ann = _get_ensembl_gene_annotations()
    else:
        raise ValueError("gene_annotation must be 'onek1k' or 'ensembl'")
    dd.C.var = dd.C.var.join(gene_ann)
    print(f"Added annotations for {dd.C.var[GAnn.start].notna().sum()}/{dd.C.n_vars} genes")

    if max_cells_per_donor is not None:
        print(f"Keeping at most {max_cells_per_donor} random cells per donor...")
        dd = _subsample_cells_per_donor(dd, max_cells_per_donor, rng)
        print(f"After cell sampling: {dd.shape}")

    print(f"Selecting SNPs (full chr22 + {snp_sample_frac*100}% from chr 1-21)...")
    all_selected_idx = []

    chr22_idx = np.where(dd.G.var.chrom == "22")[0]
    all_selected_idx.extend(chr22_idx)
    print(f"Chromosome 22: {len(chr22_idx)} SNPs (100%)")

    for chrom in range(1, 22):
        chrom_idx = np.where(dd.G.var.chrom == str(chrom))[0]
        n_snps = max(1, int(len(chrom_idx) * snp_sample_frac))
        selected_idx = rng.choice(chrom_idx, n_snps, replace=False)
        all_selected_idx.extend(selected_idx)
        print(f"  Chromosome {chrom}: {n_snps}/{len(chrom_idx)} SNPs ({snp_sample_frac*100:.1f}%)")

    all_selected_idx = np.sort(all_selected_idx)
    dd = dd[:, all_selected_idx, :, :].copy()
    print(f"After SNP selection: {dd.shape}")

    print("Filtering genes for QTL mapping...")
    dd = _filter_genes_for_qtl_mapping(dd, qtl_window=qtl_window)
    print(f"After gene filtering: {dd.shape}")

    print("Checking the paired data again...")
    _check_positive_control_eqtl(dd)

    print(f"Saving dummy dataset to {output_path}...")

    dd = dd.copy()
    G = dd.G.copy()
    if hasattr(G.X, "compute"):
        G.X = G.X.compute()
    for key in [k for k in G.layers if k is not None]:
        del G.layers[key]
    G.uns = {"kinship": G.uns["kinship"].loc[G.obs_names, G.obs_names]}
    G.uns["dummy_onek1k"] = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "n_donors": int(dd.shape[0]),
        "max_cells_per_donor": -1 if max_cells_per_donor is None else int(max_cells_per_donor),
        "snp_sample_frac": snp_sample_frac,
        "qtl_window_bp": qtl_window,
        "gene_annotation": gene_annotation,
    }
    dd.G = G
    C = dd.C.copy()
    C.var[GAnn.start] = C.var[GAnn.start].astype("int64")
    C.var[GAnn.end] = C.var[GAnn.end].astype("int64")
    dd.C = C

    if str(output_path).endswith(".dd.h5"):
        if compression_level > 0:
            tmp_path = output_path.with_name(output_path.name + ".uncompressed.tmp")
            dd.write_h5_dd(str(tmp_path))
            _compress_h5(tmp_path, output_path, level=compression_level)
            os.remove(tmp_path)
        else:
            dd.write_h5_dd(str(output_path))
    elif str(output_path).endswith(".dd.zarr"):
        dd.write_zarr_dd(str(output_path))
    else:
        raise ValueError("Output path must end with .dd.h5 or .dd.zarr")

    print(f"Saved dataset: {output_path}")

    print("Generating dataset summary...")
    summary = {
        "generator_version": GENERATOR_VERSION,
        "n_donors": dd.shape[0],
        "n_snps": dd.shape[1],
        "n_cells": dd.shape[2],
        "n_genes": dd.shape[3],
        "n_chr22_snps": len(chr22_idx),
        "max_cells_per_donor": max_cells_per_donor,
        "snp_sample_frac": snp_sample_frac,
        "qtl_window_bp": qtl_window,
        "seed": seed,
        "format": "h5" if str(output_path).endswith(".dd.h5") else "zarr",
        "compression_level": compression_level,
        "gene_annotations_included": True,
        "gene_annotation_source": gene_annotation,
        "gene_chromosomes": "1-22 only",
    }

    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Dummy OneK1K Dataset Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(f"  Saved summary: {summary_path}")

    print("\n" + "=" * 80)
    print("✓ Dummy dataset generation complete!")
    print("=" * 80)
    print(f"\nDataset location: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"\nFinal shape: {dd.shape}")
    print(f"  Donors: {dd.shape[0]}")
    print(f"  SNPs: {dd.shape[1]}")
    print(f"  Cells: {dd.shape[2]}")
    print(f"  Genes: {dd.shape[3]} (QTL-relevant only)")
    if output_path.is_file():
        print(f"\nFile size: {output_path.stat().st_size / (1024**3):.2f} GB")
    print("\nNote: Gene annotations are pre-included!")

    return dd


def main():
    parser = argparse.ArgumentParser(description="Generate dummy OneK1K dataset for tutorials")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./dummy_onek1k_dataset.dd.h5",
        help="Output path for dummy dataset (.dd.h5 or .dd.zarr)",
    )
    parser.add_argument("--n_donors", type=int, default=300, help="Number of donors to sample")
    parser.add_argument(
        "--max_cells_per_donor", type=int, default=420, help="Max. cells kept per donor (0 keeps all cells)"
    )
    parser.add_argument("--snp_sample_frac", type=float, default=0.001, help="Fraction of SNPs to keep from chr 1-21")
    parser.add_argument(
        "--qtl_window", type=int, default=500_000, help="Window size (bp) for filtering QTL-relevant genes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--no_verify_checksum", action="store_true", help="Skip checksum verification when loading data"
    )
    parser.add_argument(
        "--config_path", type=str, default="onek1k", help="Name of a shipped config, or path to a OneK1K config file"
    )
    parser.add_argument("--data_home", type=str, default=None, help="cellink data directory (default: ~/cellink_data)")
    parser.add_argument(
        "--gene_annotation",
        choices=["onek1k", "ensembl"],
        default="onek1k",
        help="Gene coordinates: Ensembl 105 table of the OneK1K record, or current Ensembl via pybiomart",
    )
    parser.add_argument("--compression_level", type=int, default=4, help="gzip level for .dd.h5 (0 = none)")

    args = parser.parse_args()

    generate_dummy_dataset(
        output_path=args.output_path,
        n_donors=args.n_donors,
        max_cells_per_donor=args.max_cells_per_donor or None,
        snp_sample_frac=args.snp_sample_frac,
        qtl_window=args.qtl_window,
        seed=args.seed,
        verify_checksum=not args.no_verify_checksum,
        config_path=args.config_path,
        data_home=args.data_home,
        gene_annotation=args.gene_annotation,
        compression_level=args.compression_level,
    )


if __name__ == "__main__":
    main()
