import argparse
import numpy as np
from pathlib import Path
from cellink.resources import get_onek1k
from cellink._core import GAnn


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

    # Filter to chromosomes 1-22 only
    valid_chroms = [str(i) for i in range(1, 23)]
    gene_mask = dd.C.var[GAnn.chrom].isin(valid_chroms)
    print(f"    Genes on chr 1-22: {gene_mask.sum()}/{len(dd.C.var)}")

    # Filter genes within QTL mapping range of any SNP
    genes_in_range = set()

    for chrom in valid_chroms:
        # Get SNPs on this chromosome
        snp_mask = dd.G.var.chrom == chrom
        if snp_mask.sum() == 0:
            continue

        snp_positions = dd.G.var.loc[snp_mask, "pos"].values
        min_snp_pos = snp_positions.min() - qtl_window
        max_snp_pos = snp_positions.max() + qtl_window

        # Get genes on this chromosome
        gene_chrom_mask = dd.C.var[GAnn.chrom] == chrom
        genes_on_chrom = dd.C.var[gene_chrom_mask]

        # Filter genes within range
        genes_in_window = genes_on_chrom[
            (genes_on_chrom[GAnn.start] <= max_snp_pos) & (genes_on_chrom[GAnn.end] >= min_snp_pos)
        ]

        genes_in_range.update(genes_in_window.index)
        print(f"    Chr {chrom}: {len(genes_in_window)} genes in range")

    # Apply filter
    gene_filter_mask = dd.C.var.index.isin(genes_in_range)
    dd = dd[..., gene_filter_mask].copy()

    print(f"    Total genes retained: {dd.shape[3]}")

    return dd


def generate_dummy_dataset(
    output_path: str = "./dummy_onek1k.dd.h5",
    n_donors: int = 100,
    snp_sample_frac: float = 0.001,
    qtl_window: int = 500_000,
    seed: int = 42,
    verify_checksum: bool = False,
    config_path: str = "./cellink/resources/config/onek1k.yaml",
):
    """
    Generate a dummy OneK1K dataset for tutorial purposes.

    Parameters
    ----------
    output_path : str
        Path where the dummy dataset will be saved (as .dd.h5 or .dd.zarr)
    n_donors : int
        Number of donors to include (randomly sampled)
    snp_sample_frac : float
        Fraction of SNPs to keep from chromosomes 1-21 (per chromosome)
    qtl_window : int
        Window size (in bp) for filtering genes relevant to QTL mapping
    seed : int
        Random seed for reproducibility
    verify_checksum : bool
        Whether to verify checksums when loading original data
    config_path : str
        Path to the OneK1K configuration file

    Returns
    -------
    DonorData
        The generated dummy dataset
    """
    np.random.seed(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Dummy OneK1K Dataset")
    print("=" * 80)

    print("Loading full OneK1K dataset...")
    dd = get_onek1k(config_path=config_path, verify_checksum=verify_checksum)
    print(f"Original dataset shape: {dd.shape}")

    print("Adding gene annotations from Ensembl...")
    ensembl_gene_id_start_end_chr = _get_ensembl_gene_annotations()
    dd.C.var = dd.C.var.join(ensembl_gene_id_start_end_chr)
    print(f"Added annotations for {ensembl_gene_id_start_end_chr.shape[0]} genes")

    print(f"Sampling {n_donors} random donors...")
    all_donors = dd.G.obs.index.values
    if len(all_donors) > n_donors:
        selected_donors = np.random.choice(all_donors, n_donors, replace=False)
        dd = dd[selected_donors, :, :, :].copy()
    print(f"After donor sampling: {dd.shape}")

    print(f"Selecting SNPs (full chr22 + {snp_sample_frac*100}% from chr 1-21)...")
    all_selected_idx = []

    chr22_idx = np.where(dd.G.var.chrom == "22")[0]
    all_selected_idx.extend(chr22_idx)
    print(f"Chromosome 22: {len(chr22_idx)} SNPs (100%)")

    for chrom in range(1, 22):
        chrom_idx = np.where(dd.G.var.chrom == str(chrom))[0]
        n_snps = max(1, int(len(chrom_idx) * snp_sample_frac))
        selected_idx = np.random.choice(chrom_idx, n_snps, replace=False)
        all_selected_idx.extend(selected_idx)
        print(f"  Chromosome {chrom}: {n_snps}/{len(chrom_idx)} SNPs ({snp_sample_frac*100:.1f}%)")

    all_selected_idx = np.sort(all_selected_idx)
    dd = dd[:, all_selected_idx, :, :].copy()
    print(f"After SNP selection: {dd.shape}")

    print("Filtering genes for QTL mapping...")
    dd = _filter_genes_for_qtl_mapping(dd, qtl_window=qtl_window)
    print(f"After gene filtering: {dd.shape}")

    print(f"Saving dummy dataset to {output_path}...")

    dd = dd.copy()
    dd.G = dd.G.copy()
    dd.C = dd.C.copy()

    if str(output_path).endswith(".dd.h5"):
        dd.write_h5_dd(str(output_path))
    elif str(output_path).endswith(".dd.zarr"):
        dd.write_zarr_dd(str(output_path))
    else:
        raise ValueError("Output path must end with .dd.h5 or .dd.zarr")

    print(f"Saved dataset: {output_path}")

    print("Generating dataset summary...")
    summary = {
        "n_donors": dd.shape[0],
        "n_snps": dd.shape[1],
        "n_cells": dd.shape[2],
        "n_genes": dd.shape[3],
        "n_chr22_snps": len(chr22_idx),
        "snp_sample_frac": snp_sample_frac,
        "qtl_window_bp": qtl_window,
        "seed": seed,
        "format": "h5" if str(output_path).endswith(".dd.h5") else "zarr",
        "gene_annotations_included": True,
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
    print(f"\nFile size: {output_path.stat().st_size / (1024**3):.2f} GB")
    print("\nNote: Gene annotations from Ensembl are pre-included!")

    return dd


def main():
    parser = argparse.ArgumentParser(description="Generate dummy OneK1K dataset for tutorials")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./dummy_onek1k.dd.h5",
        help="Output path for dummy dataset (.dd.h5 or .dd.zarr)",
    )
    parser.add_argument("--n_donors", type=int, default=100, help="Number of donors to sample")
    parser.add_argument("--snp_sample_frac", type=float, default=0.001, help="Fraction of SNPs to keep from chr 1-21")
    parser.add_argument(
        "--qtl_window", type=int, default=500_000, help="Window size (bp) for filtering QTL-relevant genes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--no_verify_checksum", action="store_true", help="Skip checksum verification when loading data"
    )
    parser.add_argument(
        "--config_path", type=str, default="./cellink/resources/config/onek1k.yaml", help="Path to OneK1K config file"
    )

    args = parser.parse_args()

    generate_dummy_dataset(
        output_path=args.output_path,
        n_donors=args.n_donors,
        snp_sample_frac=args.snp_sample_frac,
        qtl_window=args.qtl_window,
        seed=args.seed,
        verify_checksum=not args.no_verify_checksum,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
