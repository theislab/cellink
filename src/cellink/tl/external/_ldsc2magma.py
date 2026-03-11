import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_ensembl_to_entrez_map(map_tsv: str | Path) -> pd.Series:
    """
    Load a mapping TSV with columns:
      ensembl_gene_id   entrez_id
    Returns a Series indexed by ENSG (upper, no version) with values as string Entrez IDs.
    """
    map_tsv = Path(map_tsv)
    df = pd.read_csv(map_tsv, sep="\t", dtype=str)

    # Accept a few common header variants
    col_ens = None
    for c in ["ensembl_gene_id", "ENSG", "ensembl", "gene_id"]:
        if c in df.columns:
            col_ens = c
            break
    if col_ens is None:
        raise ValueError(f"Mapping file missing Ensembl column. Found: {list(df.columns)}")

    col_ent = None
    for c in ["entrez_id", "entrezgene", "entrez", "ENTREZID", "ncbi_gene_id"]:
        if c in df.columns:
            col_ent = c
            break
    if col_ent is None:
        raise ValueError(f"Mapping file missing Entrez column. Found: {list(df.columns)}")

    ens = df[col_ens].astype(str).str.strip().str.upper().str.replace(r"\..*$", "", regex=True)
    ent = df[col_ent].astype(str).str.strip()

    m = pd.Series(ent.values, index=ens.values)
    m = m[~m.index.duplicated(keep="first")]
    return m


def genesets_dir_to_entrez_gmt(
    *,
    geneset_dir: str | Path = "ldsc_genesets",
    out_gmt: str | Path | None = None,
    ensembl_to_entrez_tsv: str | Path | None = None,
    pattern: str = "*.GeneSet",
    description: str = "S-LDSC derived gene set",
    include_control: bool = False,
    control_name: str = "Control",
    remove_version_suffix: bool = True,
    uppercase: bool = True,
    min_genes: int = 1,
    sort_genes: bool = False,
    dedup_genes: bool = True,
    drop_unmapped: bool = True,
    allow_mygene_fallback: bool = False,
    species: str = "human",
    output_basename: str = "genesets.gmt",
) -> Path:
    """
    Convert *.GeneSet -> MAGMA .gmt without ID conversion.

    Defaults:
      - reads from ./ldsc_genesets
      - writes to a sibling directory ./magma_genesets/genesets.gmt
        (magma_genesets is created if needed)
    """
    geneset_dir = Path(geneset_dir).resolve()

    # ---- Default output location: sibling magma_genesets next to ldsc_genesets ----
    if out_gmt is None:
        magma_dir = geneset_dir.parent / "magma_genesets"
        magma_dir.mkdir(parents=True, exist_ok=True)
        out_gmt = magma_dir / output_basename
    else:
        out_gmt = Path(out_gmt)
        out_gmt.parent.mkdir(parents=True, exist_ok=True)

    # ---- Find input GeneSet files ----
    files = sorted(geneset_dir.glob(pattern))
    if not include_control:
        files = [p for p in files if p.stem != control_name]
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {geneset_dir}")

    if ensembl_to_entrez_tsv is not None:
        logger.warning("`ensembl_to_entrez_tsv` is ignored. No Ensembl→Entrez conversion is performed.")
    if allow_mygene_fallback:
        logger.warning("`allow_mygene_fallback` is ignored. No Ensembl→Entrez conversion is performed.")
    if not drop_unmapped:
        logger.warning("`drop_unmapped=False` has no effect when conversion is disabled.")
    if species != "human":
        logger.warning("`species` is ignored. No Ensembl→Entrez conversion is performed.")

    def norm_ens(g: str) -> str:
        g = str(g).strip()
        if remove_version_suffix:
            g = g.split(".", 1)[0]
        if uppercase:
            g = g.upper()
        return g

    def prepare_genes(genes: list[str]) -> list[str]:
        """Normalize and optionally deduplicate/sort gene IDs without conversion."""
        genes = [norm_ens(g) for g in genes if str(g).strip()]

        if dedup_genes:
            seen = set()
            genes = [g for g in genes if not (g in seen or seen.add(g))]
        if sort_genes:
            genes = sorted(genes)

        return genes

    n_written = 0
    n_skipped = 0

    with Path(out_gmt).open("w", encoding="utf-8") as out:
        for fp in files:
            set_name = fp.stem

            with fp.open("r", encoding="utf-8") as f:
                genes = [line.strip() for line in f if line.strip()]

            output_genes = prepare_genes(genes)

            if len(output_genes) < min_genes:
                logger.warning(f"Skipping {fp.name}: only {len(output_genes)} genes after normalization")
                n_skipped += 1
                continue

            row = [set_name, description] + output_genes
            out.write("\t".join(row) + "\n")
            n_written += 1

    logger.info(f"Wrote {n_written} gene sets to {out_gmt} (skipped {n_skipped})")
    return Path(out_gmt)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    p = argparse.ArgumentParser(description="Convert .GeneSet files to MAGMA .gmt without ID conversion")
    p.add_argument(
        "--geneset_dir",
        default="ldsc_genesets",
        help="Directory containing *.GeneSet (default: ldsc_genesets)",
    )
    p.add_argument(
        "--out_gmt",
        default=None,
        help="Optional output .gmt path. If omitted, writes to sibling magma_genesets/genesets.gmt",
    )
    p.add_argument(
        "--map_tsv",
        default=None,
        help="Deprecated and ignored (no Ensembl→Entrez conversion is performed).",
    )
    p.add_argument("--include_control", action="store_true")
    p.add_argument(
        "--allow_mygene_fallback",
        action="store_true",
        help="Deprecated and ignored (no Ensembl→Entrez conversion is performed).",
    )
    p.add_argument(
        "--pattern",
        default="*.GeneSet",
        help="Glob pattern for gene set files (default: *.GeneSet)",
    )
    p.add_argument(
        "--output_basename",
        default="genesets.gmt",
        help="Output filename when using default magma_genesets directory (default: genesets.gmt)",
    )
    args = p.parse_args()

    genesets_dir_to_entrez_gmt(
        geneset_dir=args.geneset_dir,
        out_gmt=args.out_gmt,
        ensembl_to_entrez_tsv=args.map_tsv,
        include_control=args.include_control,
        allow_mygene_fallback=args.allow_mygene_fallback,
        pattern=args.pattern,
        output_basename=args.output_basename,
    )
