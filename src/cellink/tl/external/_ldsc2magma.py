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
    description: str = "S-LDSC derived gene set (Entrez)",
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
    Convert *.GeneSet (Ensembl IDs) -> MAGMA .gmt with Entrez IDs.

    Defaults:
      - reads from ./ldsc_genesets
      - writes to a sibling directory ./magma_genesets/genesets.gmt
        (magma_genesets is created if needed)

    Preferred: provide ensembl_to_entrez_tsv for offline mapping.
    Optional: allow_mygene_fallback=True to query mygene.info (needs internet).
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

    # ---- Load offline map if provided ----
    ens2ent = None
    if ensembl_to_entrez_tsv is not None:
        ens2ent = load_ensembl_to_entrez_map(ensembl_to_entrez_tsv)
        logger.info(f"Loaded Ensembl→Entrez map with {ens2ent.shape[0]} entries")

    # ---- Optional online fallback ----
    mg = None
    if allow_mygene_fallback:
        try:
            import mygene  # type: ignore
            mg = mygene.MyGeneInfo()
        except Exception as e:
            raise ImportError("allow_mygene_fallback=True requires 'mygene' (pip install mygene).") from e

    def norm_ens(g: str) -> str:
        g = str(g).strip()
        if remove_version_suffix:
            g = g.split(".", 1)[0]
        if uppercase:
            g = g.upper()
        return g

    def map_to_entrez(ens_genes: list[str]) -> list[str]:
        """Return list of Entrez IDs (strings), filtered if drop_unmapped."""
        ens_genes = [norm_ens(g) for g in ens_genes if str(g).strip()]
        if dedup_genes:
            seen = set()
            ens_genes = [g for g in ens_genes if not (g in seen or seen.add(g))]

        entrez: list[str] = []
        missing: list[str] = []

        if ens2ent is not None:
            mapped = ens2ent.reindex(ens_genes)
            for g, e in zip(ens_genes, mapped.values):
                if pd.isna(e) or str(e) in ["", "nan", "None"]:
                    missing.append(g)
                else:
                    entrez.append(str(e))
        else:
            missing = ens_genes[:]  # everything missing if no offline map

    
        # mygene fallback for missing
        if mg is not None and missing:
            res = mg.querymany(
                missing,
                scopes="ensembl.gene",
                fields="entrezgene",
                species=species,
                as_dataframe=True,
                returnall=False,
            )
            if not isinstance(res, pd.DataFrame):
                res = pd.DataFrame(res)

            # Ensure a 'query' column exists
            if "query" not in res.columns:
                res = res.reset_index()
                if "query" not in res.columns:
                    res = res.rename(columns={res.columns[0]: "query"})

            if "entrezgene" in res.columns:
                # Normalize query ids
                res["query_norm"] = (
                    res["query"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.replace(r"\..*$", "", regex=True)
                )

                # Keep only rows with an actual entrezgene value
                res = res.dropna(subset=["entrezgene"])

                # If multiple hits per query, keep the first (or you can choose another policy)
                res = res.drop_duplicates(subset=["query_norm"], keep="first")

                # Build mapping Series with unique index
                map2 = pd.Series(
                    res["entrezgene"].astype("Int64").astype(str).values,
                    index=res["query_norm"].values,
                )

                for g in missing:
                    key = norm_ens(g)
                    e = map2.get(key, None)

                    # e should now be scalar; still guard against weird cases
                    if isinstance(e, pd.Series):
                        # pick first non-null if somehow still a Series
                        e = e.dropna().iloc[0] if not e.dropna().empty else None

                    if e is None or str(e) in ["<NA>", "nan", "None", ""]:
                        if not drop_unmapped:
                            continue
                    else:
                        entrez.append(str(e))


        # De-dup / sort output
        if dedup_genes:
            seen = set()
            entrez = [g for g in entrez if not (g in seen or seen.add(g))]
        if sort_genes:
            entrez = sorted(entrez)

        return entrez

    n_written = 0
    n_skipped = 0

    with Path(out_gmt).open("w", encoding="utf-8") as out:
        for fp in files:
            set_name = fp.stem

            with fp.open("r", encoding="utf-8") as f:
                ens_genes = [line.strip() for line in f if line.strip()]

            entrez_genes = map_to_entrez(ens_genes)

            if len(entrez_genes) < min_genes:
                logger.warning(f"Skipping {fp.name}: only {len(entrez_genes)} mapped Entrez genes")
                n_skipped += 1
                continue

            row = [set_name, description] + entrez_genes
            out.write("\t".join(row) + "\n")
            n_written += 1

    logger.info(f"Wrote {n_written} gene sets to {out_gmt} (skipped {n_skipped})")
    return Path(out_gmt)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    p = argparse.ArgumentParser(description="Convert Ensembl .GeneSet files to MAGMA .gmt with Entrez IDs")
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
        help="TSV with columns ensembl_gene_id and entrez_id (offline mapping)",
    )
    p.add_argument("--include_control", action="store_true")
    p.add_argument(
        "--allow_mygene_fallback",
        action="store_true",
        help="Use mygene.info for unmapped genes (needs internet)",
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
