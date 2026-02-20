import argparse
import sys
from cellink.io._pgen import stream_pgen_to_zarr

def main() -> int:
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
    """
    )
    parser.add_argument("pgen_path", nargs="+", help="Path(s) to .pgen file(s)")
    parser.add_argument("-o", "--output", required=True, help="Output Zarr directory")
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--chunk-samples", type=int, default=4096)
    parser.add_argument("--chunk-variants", type=int, default=2048)
    parser.add_argument("--memory-limit", type=float, default=10.0)
    parser.add_argument("--compressor", choices=["zstd", "lz4", "zlib"], default="zstd")
    parser.add_argument("--compression-level", type=int, default=7, choices=range(1, 10))
    parser.add_argument("--sparse", action="store_true")

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
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())