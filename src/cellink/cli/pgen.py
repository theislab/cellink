import argparse
import sys

from cellink.io._pgen import stream_pgen_to_zarr


def main() -> int:
    """Run the cellink-pgen CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more PLINK2 PGEN genotype files into a single "
            "AnnData Zarr v3 store, streaming data in blocks to support "
            "large-scale genotype matrices."
        ),
        epilog="""
    This command:

    • Streams genotype calls directly from .pgen files
    • Concatenates multiple PGEN files column-wise (variants)
    • Requires identical sample ordering across inputs
    • Writes an AnnData-compatible Zarr v3 store
    • Supports dense or sparse storage for X

    Typical use cases:
    - Split rare/common variant files → single matrix
    - Large biobank-scale genotype conversion
    - Preparation for downstream AnnData-based workflows

    Examples:

    # Single PGEN (dense matrix)
    cellink-pgen common.pgen -o common.zarr

    # Rare variants (sparse recommended)
    cellink-pgen rare.pgen -o rare.zarr --sparse
    """,
    )
    parser.add_argument("pgen_path", nargs="+", help="Path(s) to .pgen file(s)")
    parser.add_argument("-o", "--output", required=True, help="Output Zarr directory")
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--chunk-samples", type=int, default=4096)
    parser.add_argument("--chunk-variants", type=int, default=2048)
    parser.add_argument("--memory-limit", type=float, default=10.0)
    parser.add_argument(
        "--compressor",
        choices=["auto", "zstd", "lz4", "lz4hc", "zlib"],
        default="auto",
        help="Blosc compressor (default: lz4hc for 2bit, zstd for int8)",
    )
    parser.add_argument(
        "--compression-level", type=int, default=None, choices=range(1, 10), help="Default: 5 for 2bit, 7 for int8"
    )
    parser.add_argument(
        "--genotype-encoding",
        choices=["2bit", "int8"],
        default="2bit",
        help="Dense genotype storage: 2bit (default; dosages rounded to the nearest integer) or int8 (hard calls)",
    )
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument(
        "--sparse-format",
        choices=["csr", "csc"],
        default="csc",
        help="Sparse matrix format for --sparse mode (default: csc, better for variant-wise access)",
    )

    args = parser.parse_args()

    pgen_input = args.pgen_path[0] if len(args.pgen_path) == 1 else args.pgen_path

    stream_pgen_to_zarr(
        pgen_input,
        args.output,
        max_variants=args.max_variants,
        max_samples=args.max_samples,
        chunk_samples=args.chunk_samples,
        chunk_variants=args.chunk_variants,
        memory_limit_gb=args.memory_limit,
        compressor=args.compressor,
        compression_level=args.compression_level,
        sparse=args.sparse,
        sparse_format=args.sparse_format,
        genotype_encoding=args.genotype_encoding,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
