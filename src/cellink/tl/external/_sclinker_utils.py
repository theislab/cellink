from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cellink.resources._utils import retry_with_backoff

logger = logging.getLogger(__name__)


TISSUE_CODES = {
    "BLD":  "Blood",
    "BRN":  "Brain",
    "GI":   "Colon/Intestine",
    "LNG":  "Lung",
    "LIV":  "Liver",
    "KID":  "Kidney",
    "SKIN": "Skin",
    "FAT":  "Adipose",
    "HRT":  "Heart",
}


_SCLINKER_GCS_BASE = (
    "https://storage.googleapis.com/broad-alkesgroup-public"
    "/LDSCORE/Jagadeesh_Dey_sclinker/extras"
)


def _http_download(url: str, dest: Path) -> None:
    """Download a file via HTTPS, using cellink helper if available."""
    if dest.exists():
        logger.info(f"Already exists, skipping: {dest}")
        return
    logger.info(f"Downloading {url}")
    try:
        from cellink.resources._utils import _download_file
        _download_file(url, dest, checksum=None)
    except Exception:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))


def download_sclinker_enhancer_links(
    out_dir: Union[str, Path] = "sclinker_refs",
    *,
    tissue: Optional[str] = None,
    chromosomes: List[int] = list(range(1, 23)),
) -> Dict[str, Path]:
    """
    Download the sc-linker enhancer-gene reference files from GCS.

    Downloads the two large combined files that cellink then filters by tissue:

    - ``RoadmapUABCannot_regions_to_genes.txt.gz``: all-tissue Roadmap enhancer-gene activity scores.
    - ``AllPredictions.AvgHiC.ABC0.015.minus150.withcolnames.ForABCPaper.txt.gz``: all-tissue ABC model predictions.
    - ``Roadmap_map_EID_names.txt``: Roadmap EID -> tissue name mapping.

    The per-tissue files (``Roadmap_BLD_E.txt.gz`` etc.) do not exist in
    the bucket. They are produced on-the-fly by :func:`load_roadmap_links` and
    :func:`load_abc_links` when the downloaded combined files is passed.

    Parameters
    ----------
    out_dir
        Directory to download files into.
    tissue
        Accepted for API compatibility but ignored — the combined files cover
        all tissues. Pass ``tissue`` to :func:`load_roadmap_links` /
        :func:`load_abc_links` to filter after downloading.
    chromosomes
        Unused — kept for API consistency.

    Returns
    -------
    dict with keys:
        ``"roadmap"``     → ``Path`` to ``RoadmapUABCannot_regions_to_genes.txt.gz``
        ``"abc"``         → ``Path`` to ``AllPredictions.AvgHiC.ABC0.015...txt.gz``
        ``"roadmap_eid"`` → ``Path`` to ``Roadmap_map_EID_names.txt``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "roadmap": (
            "RoadmapUABCannot_regions_to_genes.txt.gz",
            out_dir / "RoadmapUABCannot_regions_to_genes.txt.gz",
        ),
        "abc": (
            "AllPredictions.AvgHiC.ABC0.015.minus150.withcolnames.ForABCPaper.txt.gz",
            out_dir / "AllPredictions.AvgHiC.ABC0.015.minus150.withcolnames.ForABCPaper.txt.gz",
        ),
        "roadmap_eid": (
            "Roadmap_map_EID_names.txt",
            out_dir / "Roadmap_map_EID_names.txt",
        ),
    }

    downloaded: Dict[str, Path] = {}
    for key, (fname, dest) in files.items():
        _http_download(f"{_SCLINKER_GCS_BASE}/{fname}", dest)
        downloaded[key] = dest

    logger.info(f"sc-linker enhancer files in {out_dir}: {list(downloaded.keys())}")
    return downloaded


def _symlink_annots_into_ld_dir(annot_prefix: str, ld_prefix: str) -> None:
    """
    Symlink .annot.gz files from the annotations directory into the LD scores
    directory. LDSC --overlap-annot looks for {ref-ld-chr}{chrom}.annot.gz
    at the same prefix as the LD score files. Cellink writes them separately,
    so we symlink (or copy if cross-filesystem) them into place.
    """
    annot_dir = Path(annot_prefix).parent
    ld_dir    = Path(ld_prefix).parent
    stem      = Path(annot_prefix).name 

    if not annot_dir.exists():
        logger.warning(f"Annotation directory not found: {annot_dir}")
        return

    linked = 0
    for annot_file in sorted(annot_dir.glob(f"{stem}*.annot.gz")):
        link = ld_dir / annot_file.name
        if not link.exists():
            try:
                link.symlink_to(annot_file.resolve())
                linked += 1
            except OSError:
                import shutil as _shutil
                _shutil.copy2(annot_file, link)
                linked += 1
    if linked:
        logger.debug(f"Linked {linked} .annot.gz files into {ld_dir}")


def run_sclinker_heritability(
    ld_prefixes: Dict[str, Dict[str, str]],
    sumstats_files: List[str],
    ref_ld_chr: str,
    w_ld_chr: str,
    out_dir: Union[str, Path],
    *,
    annotation_prefixes: Optional[Dict[str, Dict[str, str]]] = None,
    frqfile_chr: Optional[str] = None,
    runner=None,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Run S-LDSC ``--h2 --overlap-annot`` for every (program, strategy, trait).

    **Important**: sc-linker uses ``--h2`` (standard partitioned heritability),
    not ``--h2-cts``. The ``--ref-ld-chr`` is ``baseline,{program_ldscore}``
    and ``--overlap-annot`` is always used.

    ``--overlap-annot`` requires that ``.annot.gz`` files exist at the same
    prefix as the LD score files. Cellink writes them to a separate
    ``annotations/`` directory, so this function symlinks them into the LD
    score directory automatically via ``annotation_prefixes``.

    Parameters
    ----------
    ld_prefixes
        Output of ``compute_ld_scores_for_sclinker``:
        ``{program: {strategy: ld_score_prefix}}``.
    sumstats_files
        List of munged ``.sumstats.gz`` files (use absolute paths so the path
        survives Singularity bind-mount remapping).
    ref_ld_chr
        Baseline-LD prefix (e.g. from ``get_1000genomes_ld_scores()``).
    w_ld_chr
        Regression weights prefix.
    out_dir
        Directory to write ``.log`` output files.
    annotation_prefixes
        Output of ``genescores_to_annotations``:
        ``{program: {strategy: annot_prefix}}``.
        Pass this so LDSC can find the ``.annot.gz`` files for
        ``--overlap-annot``. If None, the function infers the location by
        replacing ``ldscores`` with ``annotations`` in ``ld_prefix``.
    frqfile_chr
        Allele frequency file prefix (required with ``--overlap-annot``).
    runner
        ``LDSCRunner`` instance. If None, uses the global runner.

    Returns
    -------
    dict
        ``{program: {strategy: {trait: log_path}}}``.
    """
    try:
        from cellink.tl.external._ldsc import estimate_heritability, get_ldsc_runner
    except ImportError:
        raise ImportError("cellink._ldsc is required for run_sclinker_heritability")

    if runner is None:
        runner = get_ldsc_runner()

    if frqfile_chr is None:
        logger.warning(
            "frqfile_chr not provided. --overlap-annot requires allele frequency "
            "files. Consider passing the 1000G frq prefix from "
            "get_1000genomes_frq()."
        )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Dict[str, str]]] = {}

    for program, strategies in ld_prefixes.items():
        results[program] = {}
        safe_prog = _safe_filename(program)

        for strategy_name, ld_prefix in strategies.items():
            results[program][strategy_name] = {}

            if annotation_prefixes and program in annotation_prefixes:
                annot_prefix = annotation_prefixes[program].get(strategy_name)
            else:
                annot_prefix = (
                    ld_prefix
                    .replace("/ldscores/", "/annotations/")
                    .replace(os.sep + "ldscores" + os.sep, os.sep + "annotations" + os.sep)
                )
            if annot_prefix:
                _symlink_annots_into_ld_dir(annot_prefix, ld_prefix)

            for sumstats_file in sumstats_files:
                trait = Path(sumstats_file).stem.replace(".sumstats", "")
                prog_out = out_dir / safe_prog / _safe_filename(strategy_name)
                prog_out.mkdir(parents=True, exist_ok=True)
                out_prefix = str(prog_out / trait)

                ref_ld = f"{ref_ld_chr},{ld_prefix}"

                logger.info(f"S-LDSC: {program}/{strategy_name}/{trait}")
                estimate_heritability(
                    sumstats_file=str(Path(sumstats_file).resolve()),
                    ref_ld_chr=ref_ld,
                    w_ld_chr=w_ld_chr,
                    out_prefix=out_prefix,
                    overlap_annot=True,
                    frqfile_chr=frqfile_chr,
                    print_coefficients=True,
                    print_delete_vals=True,
                    run=True,
                    runner=runner,
                )
                results[program][strategy_name][trait] = f"{out_prefix}.log"

    return results


def download_sclinker_references(
    out_dir: Union[str, Path] = "sclinker_references",
    tissue: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Download sc-linker reference files.

    **For 1000G files (PLINK, LD scores, weights, frq, HapMap3), use the
    cellink resource helpers instead** — they download from Zenodo
    (https://zenodo.org/records/10515792) which is reliable and fast:

    .. code-block:: python

        from cellink.resources import (
            get_1000genomes_ld_scores,
            get_1000genomes_ld_weights,
            get_1000genomes_plink_files,
            get_1000genomes_frq,
            get_1000genomes_hapmap3,
        )

    This function handles only the sc-linker-specific extras (enhancer-gene
    links, gene coordinates) which are only available via Google Cloud Storage:

        ``gs://broad-alkesgroup-public/LDSCORE/Jagadeesh_Dey_sclinker/extras/``

    Parameters
    ----------
    out_dir
        Root directory for downloaded files.
    tissue
        Tissue code (e.g. ``"BLD"``). None downloads all tissues.
    download_roadmap, download_abc, download_gene_coords
        Which enhancer-gene link files to download (default: all True).
    download_bims, download_frq, download_weights, download_hapmap3, download_baseline
        Kept for API compatibility but ignored — use the cellink resource
        helpers listed above instead.
    chromosomes
        Ignored (kept for API compatibility).
    """

    return download_sclinker_enhancer_links(
        out_dir=out_dir,
        tissue=tissue,
    )


def load_roadmap_eid_map(eid_file: Union[str, Path]) -> Dict[str, str]:
    """
    Load the Roadmap EID → tissue-name mapping from ``Roadmap_map_EID_names.txt``.

    Returns a dict mapping EID (e.g. ``"E062"``) to tissue label
    (e.g. ``"Primary mononuclear cells from peripheral blood"``).
    """
    df = pd.read_csv(eid_file, sep="\t", header=None, names=["EID", "tissue"])
    return dict(zip(df["EID"].str.strip(), df["tissue"].str.strip()))


def load_roadmap_links(
    roadmap_file: Union[str, Path],
    tissue: Optional[str] = None,
    eid_map_file: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load Roadmap enhancer-gene links from the combined all-tissue file.

    The actual file in the GCS bucket is
    ``RoadmapUABCannot_regions_to_genes.txt.gz``, a single combined file
    for all tissues. Per-tissue files (``Roadmap_BLD_E.txt.gz`` etc.) do not
    exist in the bucket; cellink filters the combined file here instead.

    Parameters
    ----------
    roadmap_file
        Path to ``RoadmapUABCannot_regions_to_genes.txt.gz``, downloaded by
        :func:`download_sclinker_enhancer_links`.
    tissue
        Tissue code to filter to (e.g. ``"BLD"``). If None, returns all rows.
        Uses ``Roadmap_map_EID_names.txt`` (``eid_map_file``) to match EIDs to
        tissue names. If ``eid_map_file`` is None, the ``tissue`` filter is
        applied as a case-insensitive substring match on the EID/tissue column.
    eid_map_file
        Path to ``Roadmap_map_EID_names.txt`` for EID→tissue mapping.
        Only used when ``tissue`` is not None.

    Returns
    -------
    pd.DataFrame with columns: chr, start, end, Gene, EID, activity (score).
    """

    df = pd.read_csv(roadmap_file, sep=",", compression="infer")
    if len(df.columns) == 1:
        df = pd.read_csv(roadmap_file, sep="\t", compression="infer")
    df.columns = [c.strip() for c in df.columns]
    logger.info(
        f"Loaded {len(df):,} Roadmap links from {Path(roadmap_file).name}. "
        f"Columns: {df.columns.tolist()}"
    )

    if tissue is not None:
        tissue_upper = tissue.upper()
        tissue_keywords = {
            "BLD":  ["blood", "mononuclear", "t cell", "t-cell", "b cell", "b-cell",
                     "nk cell", "cd4", "cd8", "erythro", "hsc", "monocyte",
                     "neutrophil", "lymph"],
            "BRN":  ["brain", "neuron", "cortex", "cerebellum", "hippocampus",
                     "neural", "glia"],
            "GI":   ["colon", "intestin", "sigmoid", "rectum", "duodenum",
                     "stomach", "bowel", "gastrointestinal"],
            "LNG":  ["lung", "bronchial", "alveolar", "pulmonary"],
            "LIV":  ["liver", "hepat"],
            "KID":  ["kidney", "renal"],
            "SKIN": ["skin", "keratinocyte", "fibroblast", "melanocyte", "dermis",
                     "epiderm"],
            "FAT":  ["adipos", "fat", "adipocyte"],
            "HRT":  ["heart", "cardiac", "cardiomyo", "ventricle", "aorta"],
        }
        keywords = tissue_keywords.get(tissue_upper, [tissue_upper.lower()])

        tissue_col = next(
            (c for c in df.columns
             if c.lower() in ("tissuename", "tissue_name", "tissue", "celltype",
                              "cell_type", "biosample")),
            None
        )
        if tissue_col is not None:
            pattern = "|".join(re.escape(kw) for kw in keywords)
            mask = df[tissue_col].str.lower().str.contains(pattern, regex=True, na=False)
            df = df[mask]
            logger.info(
                f"Filtered to {len(df):,} Roadmap rows for tissue={tissue} "
                f"via column '{tissue_col}' (keywords: {keywords})"
            )

        elif eid_map_file is not None:
            eid_map = load_roadmap_eid_map(eid_map_file)
            matching_eids = {
                eid for eid, name in eid_map.items()
                if any(kw in name.lower() for kw in keywords)
            }
            eid_col = next(
                (c for c in df.columns if "eid" in c.lower() or c.upper() == "EID"),
                None
            )
            if eid_col and matching_eids:
                df = df[df[eid_col].isin(matching_eids)]
                logger.info(
                    f"Filtered to {len(df):,} Roadmap rows for tissue={tissue} "
                    f"via EID column '{eid_col}' ({len(matching_eids)} EIDs)"
                )
            else:
                logger.warning(
                    f"Could not filter Roadmap for tissue={tissue}: "
                    f"no tissue-name or EID column found. "
                    f"Columns present: {df.columns.tolist()}. "
                    "Returning unfiltered DataFrame."
                )
        else:
            logger.warning(
                f"tissue={tissue} specified but no tissue-name column found "
                "and no eid_map_file provided. Returning unfiltered DataFrame. "
                f"Columns present: {df.columns.tolist()}"
            )

    if tissue is not None and len(df) == 0:
        logger.warning(
            f"Roadmap tissue filter for tissue='{tissue}' returned 0 rows. "
            f"Using ALL rows instead. "
            f"This usually means the tissuename values don't match the keywords. "
            f"Check unique tissuename values in the file:\n"
            f"  import pandas as pd; pd.read_csv('sclinker_refs/RoadmapUABCannot_regions_to_genes.txt.gz', sep=',')[['tissuename']].drop_duplicates().head(30)"
        )
        df = pd.read_csv(roadmap_file, sep=",", compression="infer")
        if len(df.columns) == 1:
            df = pd.read_csv(roadmap_file, sep="\t", compression="infer")
        df.columns = [c.strip() for c in df.columns]

    return df


def load_abc_links(
    abc_file: Union[str, Path],
    tissue: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load ABC model enhancer-gene predictions from the combined all-tissue file.

    The actual file in the GCS bucket is
    ``AllPredictions.AvgHiC.ABC0.015.minus150.withcolnames.ForABCPaper.txt.gz``, a single combined file for all tissues. Per-tissue files do not exist.

    Parameters
    ----------
    abc_file
        Path to the combined ABC predictions file, downloaded by
        :func:`download_sclinker_enhancer_links`.
    tissue
        Tissue code to filter to (e.g. ``"BLD"``). If None, returns all rows.
        Matched against the ``CellType`` or equivalent column in the file.

    Returns
    -------
    pd.DataFrame with columns: chr, start, end, TargetGene, CellType,
    ABC.Score (and others from the predictions file).
    """
    df = pd.read_csv(abc_file, sep="\t", compression="infer")
    df.columns = [c.strip() for c in df.columns]
    logger.info(
        f"Loaded {len(df):,} ABC predictions from {Path(abc_file).name}. "
        f"Columns: {df.columns.tolist()}"
    )

    if tissue is not None:
        tissue_keywords = {
            "BLD": ["blood", "k562", "gm12878", "cd4", "cd8", "nk-cell", "monocyte"],
            "BRN": ["brain", "neuron", "astrocyte", "microglia"],
            "GI":  ["colon", "intestin", "sigmoid", "rectum", "transverse"],
            "LNG": ["lung", "bronchial", "alveolar", "imr90"],
            "LIV": ["liver", "hepg2", "hepat"],
            "KID": ["kidney"],
            "SKIN": ["skin", "keratinocyte", "fibroblast", "melanocyte", "dermis"],
            "FAT":  ["adipos", "fat"],
            "HRT":  ["heart", "cardiac", "cardiomyo"],
        }
        keywords = tissue_keywords.get(tissue.upper(), [tissue.lower()])

        cell_col = next(
            (c for c in df.columns
             if c.lower() in ("celltype", "cell_type", "tissue", "biosample")),
            None
        )
        if cell_col:
            pattern = "|".join(re.escape(kw) for kw in keywords)
            mask = df[cell_col].str.lower().str.contains(pattern, regex=True, na=False)
            df = df[mask]
            logger.info(
                f"Filtered to {len(df):,} ABC rows for tissue={tissue} "
                f"(keywords: {keywords})"
            )
        else:
            logger.warning(
                f"Could not find CellType column for tissue={tissue} filtering. "
                f"Available columns: {df.columns.tolist()[:10]}. "
                "Returning unfiltered DataFrame."
            )

    return df


_GENE_COORD_CACHE = {
    "ensembl": "gene_coord_ensembl.txt",
    "hgnc":    "gene_coord_hgnc.txt",
}
_VALID_CHRS = {str(c) for c in range(1, 23)} | {"X", "Y", "MT"}


def _query_biomart_and_write_gene_coords(data_dir: Path) -> None:
    """
    Query Ensembl BioMart once and write both ENSG and HGNC gene coord files.

    A single BioMart query retrieves both ``ensembl_gene_id`` and
    ``external_gene_name`` alongside coordinates, so both cache files are
    populated in one network round-trip.

    Requires ``pybiomart`` (``pip install pybiomart``).
    """
    try:
        from pybiomart import Server
    except ImportError as exc:
        raise ImportError(
            "pybiomart is required to generate the gene coordinate file.\n"
            "Install it with:  pip install pybiomart"
        ) from exc

    logger.info("Querying Ensembl BioMart for human gene coordinates ...")
    server  = Server(host="http://www.ensembl.org")
    dataset = retry_with_backoff(lambda: server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"])

    df = retry_with_backoff(lambda: dataset.query(attributes=[
        "ensembl_gene_id",
        "external_gene_name",
        "chromosome_name",
        "start_position",
        "end_position",
    ]))
    df.columns = ["ensembl_gene_id", "hgnc_name", "CHR", "START", "END"]

    df = df[df["CHR"].astype(str).isin(_VALID_CHRS)]
    df = df.dropna(subset=["ensembl_gene_id"])
    df = df[df["ensembl_gene_id"].str.strip() != ""]

    logger.info(f"BioMart returned {len(df):,} gene entries")
    data_dir.mkdir(parents=True, exist_ok=True)

    def _dedup(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate on GENE, keeping the widest genomic span per gene.
        BioMart returns one row per transcript, so a single gene can appear
        many times.  We collapse to one row per unique GENE name: smallest
        START and largest END across all transcripts, which gives the full
        gene body extent.  This is what LDSC uses for window-based annotations.
        """
        return (
            frame
            .groupby("GENE", as_index=False)
            .agg(CHR=("CHR", "first"), START=("START", "min"), END=("END", "max"))
            [["GENE", "CHR", "START", "END"]] 
        )

    ensg = df[["ensembl_gene_id", "CHR", "START", "END"]].copy()
    ensg.columns = ["GENE", "CHR", "START", "END"]
    ensg = _dedup(ensg)
    ensg.to_csv(data_dir / _GENE_COORD_CACHE["ensembl"], sep=" ", index=False)
    logger.info(f"Wrote {len(ensg):,} unique ENSG entries → {_GENE_COORD_CACHE['ensembl']}")

    hgnc = df[df["hgnc_name"].notna() & (df["hgnc_name"].str.strip() != "")]
    hgnc = hgnc[["hgnc_name", "CHR", "START", "END"]].copy()
    hgnc.columns = ["GENE", "CHR", "START", "END"]
    hgnc = _dedup(hgnc)
    hgnc.to_csv(data_dir / _GENE_COORD_CACHE["hgnc"], sep=" ", index=False)
    logger.info(f"Wrote {len(hgnc):,} unique HGNC entries → {_GENE_COORD_CACHE['hgnc']}")


def get_gene_annotation(
    path: Optional[Union[str, Path]] = None,
    gene_id_type: str = "hgnc",
    data_home: Optional[Union[str, Path]] = None,
    refresh: bool = False,
) -> Path:
    """
    Locate or generate the LDSC gene coordinate file.

    The ``GENE`` column in the output file must match the identifiers used in
    your gene-program index / gene-set files.  A silent mismatch produces
    all-zero annotation columns.

    Parameters
    ----------
    path
        Explicit path to an existing ``GENE CHR START END`` file.  Used
        directly if the file exists; all other arguments are ignored.
    gene_id_type : ``"ensembl"`` | ``"hgnc"``
        Which identifier to put in the ``GENE`` column:

        ``"ensembl"`` *(default)*
            Ensembl stable IDs (e.g. ``ENSG00000099338``).  Use this when
            your AnnData ``var_names`` are ENSG IDs — the typical case for
            sc-linker gene programs derived from standard scRNA-seq pipelines.

        ``"hgnc"``
            HGNC gene symbols (e.g. ``CD19``, ``FOXP3``).  Use this when
            your gene-set files contain gene names.

    data_home
        Override for the cellink data directory.
    refresh
        If True, re-query BioMart even if cached files already exist.

    Returns
    -------
    pathlib.Path
        Path to a readable gene coord file with columns ``GENE CHR START END``.

    Raises
    ------
    ValueError
        If ``gene_id_type`` is not ``"ensembl"`` or ``"hgnc"``.
    ImportError
        If ``pybiomart`` is not installed and no cached/explicit file exists.
    """
    if gene_id_type not in _GENE_COORD_CACHE:
        raise ValueError(
            f"gene_id_type must be 'ensembl' or 'hgnc', got {gene_id_type!r}"
        )

    if path is not None and Path(path).exists():
        return Path(path)

    from cellink.resources._utils import get_data_home
    data_dir = Path(get_data_home(data_home))
    cache    = data_dir / _GENE_COORD_CACHE[gene_id_type]

    if cache.exists() and not refresh:
        logger.info(f"Using cached gene coordinates ({gene_id_type}): {cache}")
        return cache

    _query_biomart_and_write_gene_coords(data_dir)
    return cache


def load_gene_annotation(
    gene_annotation_file: Optional[Union[str, Path]] = None,
    gene_id_type: str = "hgnc",
    data_home: Optional[Union[str, Path]] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Load gene TSS/TES coordinates for the 100 kb window S2G strategy.

    Calls :func:`get_gene_annotation` to locate or generate the file from
    Ensembl BioMart, then reads it into a DataFrame.

    Parameters
    ----------
    gene_annotation_file
        Explicit path to an existing coord file. Passed to
        :func:`get_gene_annotation` as ``path``.
    gene_id_type : ``"ensembl"`` | ``"hgnc"``
        Which identifier is in the ``GENE`` column — must match your data.
        Default is ``"ensembl"`` because sc-linker AnnData objects typically
        have ENSG ``var_names``.
    data_home
        Override for the cellink data directory.
    refresh
        If True, re-query BioMart even if cached files exist.

    Returns
    -------
    pd.DataFrame
        Columns: ``GENE``, ``CHR``, ``START``, ``END``.
    """
    resolved = get_gene_annotation(gene_annotation_file, gene_id_type, data_home, refresh)
    df = pd.read_csv(resolved, sep=" ")
    df.columns = [c.strip() for c in df.columns]
    logger.info(
        f"Loaded {len(df):,} gene coordinates "
        f"(gene_id_type='{gene_id_type}') from {resolved.name}"
    )
    return df


def genescores_to_abc_road_bedgraph(
    genescores: pd.DataFrame,
    roadmap_links: pd.DataFrame,
    abc_links: pd.DataFrame,
    *,
    roadmap_gene_col: Optional[str] = None,
    roadmap_activity_col: Optional[str] = None,
    abc_gene_col: Optional[str] = None,
    abc_activity_col: Optional[str] = None,
    chr_col: str = "chr",
    start_col: str = "start",
    end_col: str = "end",
    use_bedtools_for_merge: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Convert gene scores to ABC_Road bedgraphs (Roadmap ∪ ABC strategy).

    This is the primary sc-linker S2G strategy. For each program (column in
    genescores), creates a bedgraph where each interval's score is the
    weighted sum of gene scores linked to that interval:

        score(interval) = gene_score C link_activity

    The Roadmap and ABC bedgraphs are merged (union) with scores summed
    where they overlap.

    Parameters
    ----------
    genescores
        DataFrame (genes C programs) with probabilistic scores in [0, 1].
        Index must be HGNC gene symbols.
    roadmap_links
        Roadmap enhancer-gene links DataFrame (from ``load_roadmap_links``).
    abc_links
        ABC model enhancer-gene links DataFrame (from ``load_abc_links``).
    roadmap_gene_col
        Column in ``roadmap_links`` with target gene names.
    roadmap_activity_col
        Activity/weight column in ``roadmap_links``. If None, uses weight=1.
    abc_gene_col
        Column in ``abc_links`` with target gene names.
    abc_activity_col
        Activity/weight column in ``abc_links``. If None, uses weight=1.
    chr_col, start_col, end_col
        Genomic coordinate columns in link DataFrames.
    use_bedtools_for_merge
        If True and bedtools is available, uses ``bedtools merge`` to merge
        overlapping intervals (matches original pipeline exactly).
        If False, uses a pure-Python merge (faster but may differ slightly).

    Returns
    -------
    dict
        Program name → bedgraph DataFrame (chr, start, end, score).
    """

    _ROADMAP_GENE_CANDIDATES = ["Gene", "gene", "GENE", "gene_name", "GeneName",
                                 "target_gene", "TargetGene"]
    _ABC_GENE_CANDIDATES     = ["TargetGene", "target_gene", "Gene", "gene",
                                 "GENE", "GeneName", "gene_name"]

    if roadmap_gene_col is None:
        roadmap_gene_col = next(
            (c for c in _ROADMAP_GENE_CANDIDATES if c in roadmap_links.columns), None
        )
        if roadmap_gene_col is None:
            raise KeyError(
                f"Cannot find a gene-name column in roadmap_links. "
                f"Columns present: {roadmap_links.columns.tolist()}\n"
                f"Pass roadmap_gene_col='<name>' explicitly."
            )
        logger.info(f"Auto-detected Roadmap gene column: '{roadmap_gene_col}'")

    if abc_gene_col is None:
        abc_gene_col = next(
            (c for c in _ABC_GENE_CANDIDATES if c in abc_links.columns), None
        )
        if abc_gene_col is None:
            raise KeyError(
                f"Cannot find a gene-name column in abc_links. "
                f"Columns present: {abc_links.columns.tolist()}\n"
                f"Pass abc_gene_col='<name>' explicitly."
            )
        logger.info(f"Auto-detected ABC gene column: '{abc_gene_col}'")

    genescores = genescores.copy()
    genescores.index = genescores.index.str.upper()
    if genescores.index.duplicated().any():
        genescores = genescores.groupby(level=0).max()

    roadmap_links = roadmap_links.copy()
    roadmap_links["_gene_upper"] = roadmap_links[roadmap_gene_col].str.upper()

    abc_links = abc_links.copy()
    abc_links["_gene_upper"] = abc_links[abc_gene_col].str.upper()

    roadmap_activity_col_r = roadmap_activity_col 
    abc_activity_col_r     = abc_activity_col

    def _build_gene_index(
        links: pd.DataFrame,
        activity_col: Optional[str],
    ) -> Dict[str, np.ndarray]:
        """
        Return {gene_upper: array of shape (N, 4)} where columns are
        [chr_idx, start, end, weight].  chr is stored as a categorical int
        to avoid per-row string operations in the hot loop.
        """
        cols = ["_gene_upper", chr_col, start_col, end_col]
        if activity_col and activity_col in links.columns:
            cols.append(activity_col)
            weight_col = activity_col
        else:
            links = links.copy()
            links["_w"] = 1.0
            weight_col = "_w"
            cols.append("_w")

        sub = links[cols].copy()
        sub[start_col] = sub[start_col].astype(np.int64)
        sub[end_col]   = sub[end_col].astype(np.int64)
        sub[weight_col] = sub[weight_col].astype(np.float64)

        index: Dict[str, tuple] = {}
        for gene, grp in sub.groupby("_gene_upper", sort=False):
            index[gene] = (
                grp[chr_col].to_numpy(),
                grp[start_col].to_numpy(),
                grp[end_col].to_numpy(),
                grp[weight_col].to_numpy(),
            )
        return index

    logger.info("Building Roadmap gene index ...")
    roadmap_idx = _build_gene_index(roadmap_links, roadmap_activity_col_r)
    logger.info(f"  {len(roadmap_idx):,} unique genes in Roadmap index")

    logger.info("Building ABC gene index ...")
    abc_idx = _build_gene_index(abc_links, abc_activity_col_r)
    logger.info(f"  {len(abc_idx):,} unique genes in ABC index")

    bedgraphs: Dict[str, pd.DataFrame] = {}

    for program in genescores.columns:
        scores = genescores[program]
        scores = scores[scores > 0]

        chrs_r, starts_r, ends_r, scores_r = [], [], [], []
        chrs_a, starts_a, ends_a, scores_a = [], [], [], []

        for gene, gene_score in scores.items():
            if gene in roadmap_idx:
                chrs_g, starts_g, ends_g, weights_g = roadmap_idx[gene]
                chrs_r.append(chrs_g)
                starts_r.append(starts_g)
                ends_r.append(ends_g)
                scores_r.append(weights_g * gene_score)

            if gene in abc_idx:
                chrs_g, starts_g, ends_g, weights_g = abc_idx[gene]
                chrs_a.append(chrs_g)
                starts_a.append(starts_g)
                ends_a.append(ends_g)
                scores_a.append(weights_g * gene_score)

        all_chrs   = (np.concatenate(chrs_r   + chrs_a)   if chrs_r   or chrs_a   else np.array([], dtype=object))
        all_starts = (np.concatenate(starts_r + starts_a) if starts_r or starts_a else np.array([], dtype=np.int64))
        all_ends   = (np.concatenate(ends_r   + ends_a)   if ends_r   or ends_a   else np.array([], dtype=np.int64))
        all_scores = (np.concatenate(scores_r + scores_a) if scores_r or scores_a else np.array([], dtype=np.float64))

        if len(all_chrs) == 0:
            logger.warning(f"No enhancer-linked intervals for program '{program}'")
            continue

        bg = pd.DataFrame({
            "chr":   all_chrs,
            "start": all_starts,
            "end":   all_ends,
            "score": all_scores,
        })

        if use_bedtools_for_merge and shutil.which("bedtools"):
            bg = _merge_bedgraph_bedtools(bg)
        else:
            bg = _merge_bedgraph_python(bg)

        bedgraphs[program] = bg

    return bedgraphs


def genescores_to_100kb_bedgraph(
    genescores: pd.DataFrame,
    gene_annotation: pd.DataFrame,
    *,
    window_kb: int = 100,
    gene_col: str = "GENE",
    chr_col: str = "CHR",
    start_col: str = "START",
    end_col: str = "END",
    use_bedtools_for_merge: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Convert gene scores to 100kb window bedgraphs.

    For each gene with a non-zero score, creates a bedgraph interval spanning
    [TSS - window_kb, TES + window_kb]. The score is the gene's probabilistic
    score (no activity weighting).

    This is the secondary sc-linker S2G strategy, reported alongside the
    primary ABC_Road strategy.

    Parameters
    ----------
    genescores
        DataFrame (genes X programs) with probabilistic scores in [0, 1].
    gene_annotation
        Gene coordinate DataFrame from ``load_gene_annotation``.
    window_kb
        Window size in kilobases around gene body (default 100).
    gene_col, chr_col, start_col, end_col
        Column names in ``gene_annotation``.
    use_bedtools_for_merge
        If True and bedtools available, use bedtools merge.

    Returns
    -------
    dict
        Program name → bedgraph DataFrame (chr, start, end, score).
    """
    window_bp = window_kb * 1000

    genescores = genescores.copy()
    genescores.index = genescores.index.str.upper()

    ga = gene_annotation.copy()
    ga["_gene_upper"] = ga[gene_col].str.upper()
    ga = ga.set_index("_gene_upper")
    ga_chr   = ga[chr_col].to_dict()
    ga_start = ga[start_col].to_dict()
    ga_end   = ga[end_col].to_dict()

    bedgraphs: Dict[str, pd.DataFrame] = {}

    if genescores.index.duplicated().any():
        n_before = len(genescores)
        genescores = genescores.groupby(level=0).max()
        logger.debug(f"Deduplicated genescores index: {n_before} → {len(genescores)} genes")

    common_genes = genescores.index[genescores.index.isin(ga_chr)]
    if len(common_genes) == 0:
        logger.warning("No overlap between genescores genes and gene_annotation. "
                       "Check gene_id_type.")
        return bedgraphs

    gs_sub = genescores.reindex(common_genes)

    chrs_arr   = np.array([ga_chr[g]   for g in common_genes], dtype=object)
    starts_arr = np.array([max(0, int(ga_start[g]) - window_bp) for g in common_genes], dtype=np.int64)
    ends_arr   = np.array([int(ga_end[g]) + window_bp            for g in common_genes], dtype=np.int64)

    for program in gs_sub.columns:
        scores = gs_sub[program].to_numpy()
        nonzero = scores > 0
        if not nonzero.any():
            logger.warning(f"No gene coordinates found for program '{program}'")
            continue

        bg = pd.DataFrame({
            "chr":   chrs_arr[nonzero],
            "start": starts_arr[nonzero],
            "end":   ends_arr[nonzero],
            "score": scores[nonzero],
        })

        if use_bedtools_for_merge and shutil.which("bedtools"):
            bg = _merge_bedgraph_bedtools(bg)
        else:
            bg = _merge_bedgraph_python(bg)

        bedgraphs[program] = bg

    return bedgraphs


def bedgraph_to_snp_annotation(
    bedgraph: Union[pd.DataFrame, str, Path],
    bim_file: Union[str, Path],
    out_prefix: str,
    *,
    use_bedtools: bool = True,
) -> Path:
    """
    Convert a bedgraph file to an S-LDSC annotation file (.annot.gz).

    Implements the logic of GSSG's ``bedgraph_to_annot.py``. Each SNP in the
    BIM file is assigned the score of the bedgraph interval it falls within
    (0 if not covered).

    Parameters
    ----------
    bedgraph
        Bedgraph DataFrame with columns (chr, start, end, score), or path to file.
    bim_file
        PLINK BIM file defining SNP positions.
    out_prefix
        Output prefix; the file ``{out_prefix}.annot.gz`` is written.
    use_bedtools
        If True and bedtools is available, use bedtools intersect (more robust).
        Otherwise falls back to pure-Python interval lookup.

    Returns
    -------
    Path
        Path to the written annotation file.
    """
    if isinstance(bedgraph, (str, Path)):
        bg = pd.read_csv(bedgraph, sep="\t", header=None,
                         names=["chr", "start", "end", "score"])
    else:
        bg = bedgraph.copy()

    bim = pd.read_csv(
        bim_file, sep=r"\s+", header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
    )

    out_path = Path(f"{out_prefix}.annot.gz")

    if use_bedtools and shutil.which("bedtools"):
        annot_values = _annotate_with_bedtools(bg, bim)
    else:
        annot_values = _annotate_python(bg, bim)

    annot_df = pd.DataFrame({"ANNOT": annot_values.astype(np.float64)})
    annot_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    n_nonzero = int((annot_values > 0).sum())
    logger.info(f"Wrote {out_path}: {n_nonzero:,} / {len(bim):,} SNPs annotated")
    return out_path


def genescores_to_annotations(
    genescores: pd.DataFrame,
    roadmap_links: pd.DataFrame,
    abc_links: pd.DataFrame,
    gene_annotation: pd.DataFrame,
    bim_prefix: str,
    out_dir: Union[str, Path],
    *,
    tissue: str = "BLD",
    chromosomes: List[int] = list(range(1, 23)),
    window_kb: int = 100,
    save_bedgraphs: bool = True,
    use_bedtools: bool = True,
    **link_kwargs,
) -> Dict[str, Dict[str, str]]:
    """
    Full Step 2: gene scores → bedgraphs → per-chromosome SNP annotations.

    Produces TWO annotation strategies per program, matching the sc-linker paper:
    - ``ABC_Road_{tissue}``: Roadmap ∪ ABC enhancer-gene links (weighted)
    - ``100kb``: gene body ± window_kb (unweighted)

    Parameters
    ----------
    genescores
        DataFrame (genes X programs) with probabilistic scores in [0, 1].
    roadmap_links
        Roadmap enhancer-gene links (from ``load_roadmap_links``).
    abc_links
        ABC enhancer-gene links (from ``load_abc_links``).
    gene_annotation
        Gene coordinate file (from ``load_gene_annotation``).
    bim_prefix
        Prefix for per-chromosome BIM files.
        Example: ``"refs/1000G.EUR.QC."`` → files ``...1.bim``, ``...2.bim``, ...
    out_dir
        Root output directory. Creates sub-dirs per program and strategy.
    tissue
        Tissue code (used only for naming the strategy folder).
    chromosomes
        Chromosomes to process.
    window_kb
        Window for 100kb strategy.
    save_bedgraphs
        Whether to save bedgraph files alongside annotation files.
    use_bedtools
        Use bedtools binary for interval merging and annotation.

    Returns
    -------
    dict
        ``{program_name: {"ABC_Road": annot_prefix, "100kb": annot_prefix}}``
        where annot_prefix is a string like ``"out_dir/program/ABC_Road_BLD/program."``
        (without the chromosome number, to be passed to LDSC).
    """
    out_dir = Path(out_dir).resolve() 
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing ABC_Road bedgraphs...")
    abc_road_bgs = genescores_to_abc_road_bedgraph(
        genescores, roadmap_links, abc_links,
        use_bedtools_for_merge=use_bedtools, **link_kwargs
    )

    logger.info("Computing 100kb bedgraphs...")
    kb100_bgs = genescores_to_100kb_bedgraph(
        genescores, gene_annotation,
        window_kb=window_kb,
        use_bedtools_for_merge=use_bedtools,
    )

    annotation_prefixes: Dict[str, Dict[str, str]] = {}

    for program in genescores.columns:
        safe_name = _safe_filename(program)
        prog_dir = out_dir / safe_name
        prog_dir.mkdir(exist_ok=True)

        annotation_prefixes[program] = {}

        for strategy_name, bgs in [
            (f"ABC_Road_{tissue}", abc_road_bgs),
            ("100kb", kb100_bgs),
        ]:
            if program not in bgs:
                logger.warning(f"No bedgraph for '{program}' / {strategy_name}, skipping")
                continue

            bg_df = bgs[program]
            strategy_dir = prog_dir / strategy_name
            strategy_dir.mkdir(exist_ok=True)

            if save_bedgraphs:
                bg_path = strategy_dir / f"{safe_name}.bedgraph"
                bg_df.to_csv(bg_path, sep="\t", header=False, index=False)

            for chrom in chromosomes:
                bim_file = Path(f"{bim_prefix}{chrom}.bim")
                if not bim_file.exists():
                    logger.debug(f"BIM file not found for chr{chrom}: {bim_file}")
                    continue

                chrom_str = str(chrom)
                bg_chrom = bg_df[
                    bg_df["chr"].astype(str).str.replace("^chr", "", regex=True) == chrom_str
                ]

                if len(bg_chrom) == 0:
                    _write_zero_annotation(bim_file, str(strategy_dir / f"{safe_name}.{chrom}"))
                    continue

                bedgraph_to_snp_annotation(
                    bg_chrom, bim_file,
                    out_prefix=str(strategy_dir / f"{safe_name}.{chrom}"),
                    use_bedtools=use_bedtools,
                )

            annot_prefix = str(strategy_dir / f"{safe_name}.")
            annotation_prefixes[program][strategy_name] = annot_prefix
            logger.info(f"  {program}/{strategy_name}: annotations at {annot_prefix}*.annot.gz")

    return annotation_prefixes


def compute_ld_scores_for_sclinker(
    annotation_prefixes: Dict[str, Dict[str, str]],
    bim_prefix: str,
    ld_scores_dir: Union[str, Path],
    *,
    hapmap3_snps_prefix: Optional[str] = None,
    hapmap3_snps_file: Optional[Union[str, Path]] = None,
    chromosomes: List[int] = list(range(1, 23)),
    n_jobs: int = 4,
    runner=None,
) -> Dict[str, Dict[str, str]]:
    """
    Compute LD scores for all sc-linker annotations.

    Calls the existing cellink ``compute_ld_scores_with_annotations_from_bimfile``
    for each (program, strategy, chromosome) combination.

    Chromosome jobs are dispatched in parallel using a thread pool, each job
    is an independent Singularity/subprocess call so threads work well.

    Parameters
    ----------
    annotation_prefixes
        Output of ``genescores_to_annotations``.
    bim_prefix
        Prefix for per-chromosome BIM/PLINK files.
    ld_scores_dir
        Directory to write LD score files.
    hapmap3_snps_file
        Path to a single HapMap3 SNP list file (e.g. ``hm3_no_MHC.list.txt``
        from ``get_1000genomes_hapmap3()``). Used as ``--print-snps`` for
        every chromosome. Takes precedence over ``hapmap3_snps_prefix``.
    hapmap3_snps_prefix
        Prefix for per-chromosome HapMap3 SNP files (e.g. ``"refs/hm."``).
        Use ``hapmap3_snps_file`` instead when you have a single combined file.
    chromosomes
        Chromosomes to process. For a quick tutorial run use ``[21, 22]``.
    n_jobs
        Number of chromosomes to process in parallel. Each job is one
        Singularity subprocess call; ``n_jobs=4`` is a safe default on an
        HPC node. Set to 1 to disable parallelism.
    runner
        cellink LDSCRunner. If None, uses global runner.

    Returns
    -------
    dict
        ``{program: {strategy: ld_score_prefix}}``
        where ld_score_prefix ends in ``.`` (chromosome appended by LDSC).
    """
    import concurrent.futures

    try:
        from cellink.tl.external._ldsc import (
            compute_ld_scores_with_annotations_from_bimfile,
            get_ldsc_runner,
        )
    except ImportError:
        raise ImportError("cellink LDSC wrappers required")

    if runner is None:
        runner = get_ldsc_runner()

    ld_scores_dir = Path(ld_scores_dir).resolve()
    ld_scores_dir.mkdir(parents=True, exist_ok=True)

    print_snps_global: Optional[str] = None
    if hapmap3_snps_file and Path(hapmap3_snps_file).exists():
        print_snps_global = str(hapmap3_snps_file)

    Job = tuple  # (program, strategy_name, annot_prefix, chrom, out_prefix, print_snps)
    jobs: List[Job] = []
    ld_prefixes: Dict[str, Dict[str, str]] = {}

    for program, strategies in annotation_prefixes.items():
        ld_prefixes[program] = {}
        for strategy_name, annot_prefix in strategies.items():
            safe_name = _safe_filename(program)
            strategy_ld_dir = ld_scores_dir / safe_name / strategy_name
            strategy_ld_dir.mkdir(parents=True, exist_ok=True)
            ld_prefixes[program][strategy_name] = str(strategy_ld_dir / f"{safe_name}.")

            for chrom in chromosomes:
                annot_file = f"{annot_prefix}{chrom}.annot.gz"
                if not Path(annot_file).exists():
                    logger.debug(f"Annotation missing: {annot_file}")
                    continue

                bim_file  = f"{bim_prefix}{chrom}"
                out_prefix = str(strategy_ld_dir / f"{safe_name}.{chrom}")

                print_snps = print_snps_global
                if print_snps is None and hapmap3_snps_prefix:
                    snp_file = f"{hapmap3_snps_prefix}{chrom}.snp"
                    if Path(snp_file).exists():
                        print_snps = snp_file

                jobs.append((program, strategy_name, chrom, annot_file,
                             bim_file, out_prefix, print_snps))

    n_total = len(jobs)
    logger.info(
        f"Computing LD scores: {n_total} jobs "
        f"({len(annotation_prefixes)} programs X "
        f"{sum(len(s) for s in annotation_prefixes.values()) // max(len(annotation_prefixes),1)} strategies X "
        f"{len(chromosomes)} chromosomes), n_jobs={n_jobs}"
    )

    def _run_one(job: Job) -> str:
        program, strategy_name, chrom, annot_file, bim_file, out_prefix, print_snps = job
        compute_ld_scores_with_annotations_from_bimfile(
            bfile_prefix=bim_file,
            annot_file=annot_file,
            out_prefix=out_prefix,
            print_snps=print_snps,
            runner=runner,
        )
        return f"{program}/{strategy_name}/chr{chrom}"

    errors: List[str] = []
    completed = 0

    if n_jobs == 1:
        for job in jobs:
            try:
                label = _run_one(job)
                completed += 1
                logger.info(f"  [{completed}/{n_total}] done: {label}")
            except Exception as exc:
                errors.append(f"{job[0]}/{job[1]}/chr{job[2]}: {exc}")
                logger.error(f"  FAILED: {errors[-1]}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            future_to_job = {pool.submit(_run_one, job): job for job in jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    label = future.result()
                    completed += 1
                    logger.info(f"  [{completed}/{n_total}] done: {label}")
                except Exception as exc:
                    errors.append(f"{job[0]}/{job[1]}/chr{job[2]}: {exc}")
                    logger.error(f"  FAILED: {errors[-1]}")

    if errors:
        logger.warning(
            f"{len(errors)} LD score job(s) failed:\n" + "\n".join(errors)
        )

    return ld_prefixes


def load_sclinker_heritability_results(
    results_dir: Union[str, Path],
    *,
    log_pattern: str = "**/*.log",
) -> pd.DataFrame:
    """
    Parse all S-LDSC .log files from a sc-linker run into a DataFrame.

    Extracts Enrichment, Enrichment_std_error, Coefficient, and tau*
    for the annotation of interest (last non-baseline annotation column).

    Parameters
    ----------
    results_dir
        Directory containing .log files (searched recursively).
    log_pattern
        Glob pattern for log files.

    Returns
    -------
    DataFrame with columns:
        program, strategy, trait, Enrichment, Enrichment_std_error,
        Enrichment_z_score, Coefficient, Coefficient_std_error, tau_star
    """
    results_dir = Path(results_dir)
    log_files = sorted(results_dir.glob(log_pattern))

    if not log_files:
        raise FileNotFoundError(f"No log files matching '{log_pattern}' in {results_dir}")

    logger.info(f"Found {len(log_files)} log files in {results_dir}")

    rows = []
    skipped = []
    for log_file in log_files:
        parts = log_file.relative_to(results_dir).parts
        trait = log_file.stem
        try:
            strategy = parts[-2]
            program  = parts[-3]
        except IndexError:
            program = strategy = "unknown"

        row = _parse_ldsc_log(log_file)
        if row:
            row.update({"program": program, "strategy": strategy, "trait": trait})
            rows.append(row)
        else:
            skipped.append(str(log_file))

    if skipped:
        logger.warning(
            f"{len(skipped)} log file(s) could not be parsed "
            f"(--overlap-annot may have failed; check the logs):\n"
            + "\n".join(f"  {p}" for p in skipped[:5])
            + ("\n  ..." if len(skipped) > 5 else "")
        )

    if not rows:
        logger.warning(
            "No results parsed. Possible causes:\n"
            "  1. No .results files alongside the .log files — check that LDSC ran with "
            "--overlap-annot and that annotation_prefixes= was passed to run_sclinker_heritability().\n"
            "  2. LDSC exited before writing .results — open a .log file and check for errors."
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(
        f"Loaded {len(df)} results: "
        f"{df['program'].nunique()} programs, {df['trait'].nunique()} traits"
    )
    return df


def compute_escore(
    results_df: pd.DataFrame,
    control_program: str = "AllCoding",
    *,
    control_strategy: Optional[str] = None,
    enrichment_col: str = "Enrichment",
    se_col: str = "Enrichment_std_error",
) -> pd.DataFrame:
    """
    Compute the E-score: enrichment of a program minus the all-coding control.

    E-score(program, trait) = Enrichment(program) - Enrichment(AllCoding_control)
    SE_E = sqrt(SE_program² + SE_control²)
    z_E  = E-score / SE_E

    Parameters
    ----------
    results_df
        Output of ``load_sclinker_heritability_results``.
    control_program
        Name of the all-coding control program.
    control_strategy
        Strategy to use for the control (if None, matches current strategy).
    enrichment_col, se_col
        Column names for enrichment and its standard error.

    Returns
    -------
    DataFrame with added columns E_score, E_score_se, E_score_z.
    """
    df = results_df.copy()

    ctrl_mask = df["program"] == control_program
    if control_strategy:
        ctrl_mask &= df["strategy"] == control_strategy

    # When multiple strategies exist per control program, take one control row
    # per (trait, strategy) pair; fall back to any matching trait if unambiguous.
    ctrl = (
        df[ctrl_mask]
        .groupby(["trait", "strategy"])[[enrichment_col, se_col]]
        .first()
        .rename(columns={enrichment_col: "_ctrl_enr", se_col: "_ctrl_se"})
    )

    if ctrl.empty:
        logger.warning(f"Control program '{control_program}' not found; skipping E-score")
        return df

    df = df.copy()

    # Merge on (trait, strategy) so each row gets the correct per-strategy control
    merged = df.merge(
        ctrl.reset_index(),
        on=["trait", "strategy"],
        how="left",
    )
    merged["E_score"]    = merged[enrichment_col] - merged["_ctrl_enr"]
    merged["E_score_se"] = np.sqrt(merged[se_col] ** 2 + merged["_ctrl_se"] ** 2)
    merged["E_score_z"]  = merged["E_score"] / (merged["E_score_se"] + 1e-12)
    return merged.drop(columns=["_ctrl_enr", "_ctrl_se"])



_PARSE_PY_PATCH_MARKER = "chr_ld[0].columns"
_PARSE_PY_BUGGY_PATTERNS = [
    (
        "x = pd.concat(chr_ld)  # automatically sorted by chromosome",
        "x = pd.concat(chr_ld)[chr_ld[0].columns]  # automatically sorted by chromosome",
    ),
    (
        "x = pd.concat(chr_ld)",
        "x = pd.concat(chr_ld)[chr_ld[0].columns]",
    ),
]


def _get_parse_py_path(runner) -> Optional[str]:
    """
    Return the path to ``ldscore/parse.py`` to be used for reading/writing.

    Resolution order:

    1. ``runner.parse_script`` / ``runner.config["parse_script"]`` — the
       canonical field on ``LDSCRunner``.  Always set this explicitly for
       Docker and Singularity; the path inside the container is typically
       ``/ldsc/ldscore/parse.py``.
    2. Auto-discovery via PATH — find ``ldsc.py`` on the host PATH and infer
       the sibling ``ldscore/parse.py``.  Works for local installs; will not
       work inside containers.

    Returns None if the path cannot be determined.
    """
    explicit = getattr(runner, "parse_script", None) or runner.config.get("parse_script")
    if explicit:
        return str(explicit)
    
    ldsc_cmd = runner.config.get("ldsc_command", "ldsc.py")
    ldsc_bin = shutil.which(ldsc_cmd)
    if ldsc_bin is None:
        return None

    for candidate in [
        Path(ldsc_bin).parent / "ldscore" / "parse.py",
        Path(ldsc_bin).parent.parent / "ldscore" / "parse.py",
    ]:
        if candidate.exists():
            return str(candidate)

    return None


def _read_parse_py_via_runner(runner) -> Optional[str]:
    """
    Read the source of ``ldscore/parse.py`` through the configured runner.

    - **local**: reads the file directly from disk.
    - **docker**: uses ``docker run --rm ... cat <path>``.
    - **singularity**: uses ``singularity exec ... cat <path>``.

    Returns the source text, or None if the file cannot be read.
    """
    mode      = runner.config.get("execution_mode", "local")
    parse_path = _get_parse_py_path(runner)

    if parse_path is None:
        logger.warning(
            "Cannot locate ldscore/parse.py. "
            "Set parse_script in your runner config to the explicit path, "
            "e.g. /ldsc/ldscore/parse.py for Singularity/Docker."
        )
        return None

    if mode == "local":
        p = Path(parse_path)
        return p.read_text() if p.exists() else None

    elif mode == "docker":
        image  = runner.config.get("docker_image", "zijingliu/ldsc")
        result = subprocess.run(
            ["docker", "run", "--rm", image, "cat", parse_path],
            capture_output=True, text=True,
        )
        return result.stdout if result.returncode == 0 else None

    elif mode == "singularity":
        image  = runner.config.get("singularity_image", "")
        result = subprocess.run(
            ["singularity", "exec", image, "cat", parse_path],
            capture_output=True, text=True,
        )
        return result.stdout if result.returncode == 0 else None

    return None


def _write_parse_py_via_runner(runner, patched_source: str) -> bool:
    """
    Write ``patched_source`` back to ``ldscore/parse.py`` through the runner.

    **local**
        Overwrites ``parse.py`` on disk; backs up original as ``parse.py.bak``.

    **docker**
        ``docker cp`` + ``docker commit`` — the patched file is baked
        permanently into the image. Idempotent and survives restarts.

    **singularity**
        SIF images are read-only squashfs archives — they cannot be edited
        in-place.  cellink supports three strategies, chosen via
        ``singularity_patch_strategy`` in the runner config:

        ``"overlay"`` *(default — HPC-friendly, no root needed)*
            Creates a persistent ext3 overlay image alongside the SIF.
            On every ``ldsc.py`` call cellink appends
            ``--overlay <overlay.img>`` so the patch is always active.
            The overlay file lives at ``singularity_overlay_path`` (default:
            ``~/.cellink/ldsc_overlay.img``).

        ``"sandbox"``
            Converts the SIF to a writable directory sandbox once, then
            patches ``parse.py`` inside it.  Subsequent runs use
            ``singularity exec <sandbox_dir>`` instead of the SIF.  The
            sandbox path is stored in ``singularity_sandbox_path`` (default:
            next to the SIF as ``<sif>.sandbox/``).

        ``"rebuild"``
            Converts to sandbox, patches, then rebuilds a new SIF.  The
            patched SIF replaces the original (original is backed up as
            ``<sif>.bak.sif``).  Requires Singularity build privileges (or
            ``--fakeroot``).  After rebuild the runner ``singularity_image``
            config is updated to point at the new image.

    Returns True on success, False otherwise.
    """
    import tempfile
    mode       = runner.config.get("execution_mode", "local")
    parse_path = _get_parse_py_path(runner)

    if parse_path is None:
        return False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_parse_patched.py", delete=False
    ) as tf:
        tf.write(patched_source)
        tmp_path = tf.name

    try:
        if mode == "local":
            p = Path(parse_path)
            if not p.exists():
                logger.error(f"parse.py not found on disk at {p}")
                return False
            shutil.copy2(str(p), str(p) + ".bak")
            shutil.copy2(tmp_path, str(p))
            logger.info(f"Patched {p}  (backup: {p}.bak)")
            return True

        elif mode == "docker":
            image = runner.config.get("docker_image", "zijingliu/ldsc")
            cid_result = subprocess.run(
                ["docker", "create", image],
                capture_output=True, text=True,
            )
            if cid_result.returncode != 0:
                logger.error(f"docker create failed: {cid_result.stderr}")
                return False
            cid = cid_result.stdout.strip()
            try:
                subprocess.run(
                    ["docker", "cp", tmp_path, f"{cid}:{parse_path}"],
                    check=True, capture_output=True,
                )
                subprocess.run(
                    ["docker", "commit", cid, image],
                    check=True, capture_output=True,
                )
                logger.info(
                    f"Patched {parse_path} committed to Docker image '{image}' permanently."
                )
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Docker patch failed: {e.stderr}")
                return False
            finally:
                subprocess.run(["docker", "rm", cid], capture_output=True)

        elif mode == "singularity":
            strategy = runner.config.get("singularity_patch_strategy", "overlay")
            sif      = runner.config.get("singularity_image", "")

            if strategy == "overlay":
                return _singularity_patch_overlay(runner, sif, parse_path, tmp_path)
            elif strategy == "sandbox":
                return _singularity_patch_sandbox(runner, sif, parse_path, tmp_path, rebuild=False)
            elif strategy == "rebuild":
                return _singularity_patch_sandbox(runner, sif, parse_path, tmp_path, rebuild=True)
            else:
                logger.error(
                    f"Unknown singularity_patch_strategy '{strategy}'. "
                    "Choose 'overlay', 'sandbox', or 'rebuild'."
                )
                return False

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return False


def _singularity_patch_overlay(runner, sif: str, parse_path: str, patched_tmp: str) -> bool:
    """
    Singularity overlay strategy: create a persistent ext3 overlay image,
    copy the patched parse.py into it, and configure the runner to always
    mount it.

    The overlay file path defaults to ``~/.cellink/ldsc_overlay.img`` but
    can be overridden via ``singularity_overlay_path`` in the runner config.

    On every subsequent call, ``run_command`` in BaseToolRunner automatically
    appends ``--overlay <overlay.img>`` to the ``singularity exec`` invocation
    because ``_ldsc_overlay_path`` is stored in ``runner.config``.
    """
    overlay_path = runner.config.get(
        "singularity_overlay_path",
        str(Path.home() / ".cellink" / "ldsc_overlay.img"),
    )
    overlay_path = str(overlay_path)
    overlay_size_mb = runner.config.get("singularity_overlay_size_mb", 256)

    if not Path(overlay_path).exists():
        logger.info(f"Creating Singularity overlay image: {overlay_path} ({overlay_size_mb} MB)")
        Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["singularity", "overlay", "create",
             "--size", str(overlay_size_mb), overlay_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error(f"singularity overlay create failed: {result.stderr}")
            return False

    result = subprocess.run(
        [
            "singularity", "exec",
            "--overlay", f"{overlay_path}:rw",
            sif,
            "bash", "-c",
            f"mkdir -p $(dirname {parse_path}) && cp /dev/stdin {parse_path}",
        ],
        input=open(patched_tmp).read(),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"Failed to write patched parse.py into overlay: {result.stderr}")
        return False

    runner.config["_ldsc_overlay_path"] = overlay_path
    logger.info(
        f"Patched {parse_path} written into overlay {overlay_path}. "
        "Overlay is mounted automatically on every ldsc.py call (permanent "
        "across sessions as long as the overlay file exists)."
    )
    return True


def _singularity_patch_sandbox(
    runner, sif: str, parse_path: str, patched_tmp: str, rebuild: bool
) -> bool:
    """
    Singularity sandbox strategy.

    1. Convert the SIF to a writable sandbox directory (once).
    2. Copy the patched ``parse.py`` into the sandbox.
    3a. If ``rebuild=False``: update ``runner.config["singularity_image"]``
        to point at the sandbox directory so all subsequent ``singularity exec``
        calls use it directly.
    3b. If ``rebuild=True``: rebuild a new SIF from the sandbox, back up the
        original SIF, and update ``runner.config["singularity_image"]`` to the
        new SIF.
    """
    sandbox_path = runner.config.get(
        "singularity_sandbox_path",
        str(Path(sif).with_suffix("")) + ".sandbox",
    )

    if not Path(sandbox_path).exists():
        logger.info(f"Converting {sif} to writable sandbox at {sandbox_path} ...")
        result = subprocess.run(
            ["singularity", "build", "--sandbox", sandbox_path, sif],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning("sandbox build failed without root; retrying with --fakeroot")
            result = subprocess.run(
                ["singularity", "build", "--fakeroot", "--sandbox", sandbox_path, sif],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.error(f"singularity build --sandbox failed: {result.stderr}")
                return False

    sandbox_parse = Path(sandbox_path) / parse_path.lstrip("/")
    sandbox_parse.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(patched_tmp, str(sandbox_parse))
    logger.info(f"Patched {sandbox_parse}")

    if not rebuild:
        runner.config["singularity_image"] = sandbox_path
        logger.info(
            f"Runner updated: singularity_image → {sandbox_path}. "
            "All subsequent ldsc.py calls will use the patched sandbox."
        )
        return True
    else:
        new_sif = str(Path(sif).with_suffix("")) + ".patched.sif"
        backup_sif = sif + ".bak.sif"
        logger.info(f"Rebuilding SIF from sandbox: {sandbox_path} → {new_sif} ...")
        result = subprocess.run(
            ["singularity", "build", new_sif, sandbox_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["singularity", "build", "--fakeroot", new_sif, sandbox_path],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.error(f"singularity build from sandbox failed: {result.stderr}")
                return False

        shutil.move(sif, backup_sif)
        shutil.move(new_sif, sif)
        runner.config["singularity_image"] = sif
        logger.info(
            f"Rebuilt SIF: {sif}  (original backed up as {backup_sif}). "
            "The patch is permanently baked into the image."
        )
        return True


def check_and_patch_ldsc_parse_bug(runner) -> dict:
    """
    Check whether the LDSC installation has the pandas column-sort bug
    (ldsc issue `#342 <https://github.com/bulik/ldsc/issues/342>`_ /
    PR `#341 <https://github.com/bulik/ldsc/pull/341>`_) and patch it.

    The bug: ``pd.concat`` in ``ldscore/parse.py`` alphabetically re-sorts
    annotation columns after ``pd.concat`` across chromosomes in ``ldscore/parse.py``
    (line ~147), causing ``_check_variance()`` to raise an ``IndexError``
    during ``--h2 --overlap-annot``.

    The fix (PR #341): reindex the concatenated DataFrame to the column order
    of the first chromosome file: ``pd.concat(chr_ld)[chr_ld[0].columns]``.

    Works for **local**, **Docker** and **Singularity** execution modes.

    **Configuration** — add ``parse_script`` to your runner config so cellink
    knows exactly where ``parse.py`` lives, especially for containerised setups:

    .. code-block:: python

        runner = configure_ldsc_runner(config_dict={
            "execution_mode":  "singularity",
            "singularity_image": "/path/to/ldsc.sif",
            "ldsc_command":    "ldsc.py",
            "parse_script":   "/ldsc/ldscore/parse.py",  # ← explicit path
        })

    If ``parse_script`` is omitted, cellink tries to auto-discover the path
    from ``PATH`` (works for local installs; may fail for containers).

    Behaviour by mode
    -----------------
    - **local** — patches ``parse.py`` on disk and backs up the original as
      ``parse.py.bak``.
    - **docker** — patches the file inside the image via ``docker cp`` +
      ``docker commit``.  The image is updated in-place and the patch survives
      container restarts.
    - **singularity** — SIF images are read-only, so cellink writes the patched
      ``parse.py`` to a host-side shadow directory (default:
      ``~/.cellink/ldsc_patch/ldscore/``) and injects
      ``PYTHONPATH=<shadow>:$PYTHONPATH`` into every subsequent ``ldsc.py``
      call.  Set ``ldsc_patch_shadow_dir`` in the runner config to override
      the shadow location.

    Parameters
    ----------
    runner
        A configured ``LDSCRunner`` instance.

    Returns
    -------
    dict with keys:
        ``"status"``     — ``"already_patched"``, ``"patched"``,
                           ``"patch_failed"`` or ``"not_found"``
        ``"mode"``       — ``"local"``, ``"docker"`` or ``"singularity"``
        ``"parse_path"`` — resolved path to ``parse.py`` (or None)
        ``"detail"``     — human-readable explanation
    """
    mode       = runner.config.get("execution_mode", "local")
    parse_path = _get_parse_py_path(runner)
    source     = _read_parse_py_via_runner(runner)

    if source is None:
        return {
            "status":     "not_found",
            "mode":       mode,
            "parse_path": parse_path,
            "detail": (
                "Could not read ldscore/parse.py. "
                "Add parse_script to your runner config with the explicit path, "
                "e.g. '/ldsc/ldscore/parse.py' for Singularity/Docker."
            ),
        }

    if _PARSE_PY_PATCH_MARKER in source:
        return {
            "status":     "already_patched",
            "mode":       mode,
            "parse_path": parse_path,
            "detail":     "parse.py already patched (chr_ld[0].columns reindex present) — no action needed.",
        }

    patched = source
    applied = False
    for old, new in _PARSE_PY_BUGGY_PATTERNS:
        if old in patched:
            patched = patched.replace(old, new)
            applied = True

    if not applied:
        return {
            "status":     "patch_failed",
            "mode":       mode,
            "parse_path": parse_path,
            "detail": (
                "Found parse.py but could not locate the expected pd.concat "
                "call. The installed LDSC version may differ. "
                "Apply the fix from https://github.com/bulik/ldsc/pull/341 manually."
            ),
        }

    success = _write_parse_py_via_runner(runner, patched)
    if success:
        return {
            "status":     "patched",
            "mode":       mode,
            "parse_path": parse_path,
            "detail": (
                f"Applied chr_ld[0].columns reindex fix to {parse_path} "
                f"(ldsc issue #342 / PR #341) via {mode}."
            ),
        }
    return {
        "status":     "patch_failed",
        "mode":       mode,
        "parse_path": parse_path,
        "detail": (
            "Found and modified parse.py source but could not write it back. "
            "Check file permissions (local) or container access (Docker/Singularity)."
        ),
    }


def _merge_bedgraph_bedtools(bg: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping intervals using bedtools merge, summing scores."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".bed", delete=False
    ) as f_in:
        tmp_in = f_in.name
        # Sort by chr, start
        bg_sorted = bg.sort_values(["chr", "start"]).reset_index(drop=True)
        bg_sorted.to_csv(f_in, sep="\t", header=False, index=False)

    with tempfile.NamedTemporaryFile(
        mode="r", suffix=".bed", delete=False
    ) as f_out:
        tmp_out = f_out.name

    try:
        sort_result = subprocess.run(
            ["bedtools", "sort", "-i", tmp_in],
            capture_output=True, text=True, check=True,
        )
        merge_result = subprocess.run(
            ["bedtools", "merge", "-i", "stdin", "-c", "4", "-o", "sum"],
            input=sort_result.stdout,
            capture_output=True, text=True, check=True,
        )
        rows = []
        for line in merge_result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            rows.append({
                "chr": parts[0],
                "start": int(parts[1]),
                "end": int(parts[2]),
                "score": float(parts[3]),
            })
        return pd.DataFrame(rows, columns=["chr", "start", "end", "score"])
    except subprocess.CalledProcessError as e:
        logger.warning(f"bedtools merge failed: {e.stderr}. Falling back to Python merge.")
        return _merge_bedgraph_python(bg)
    finally:
        for f in [tmp_in, tmp_out]:
            try:
                os.unlink(f)
            except Exception:
                pass


def _merge_bedgraph_python(bg: pd.DataFrame) -> pd.DataFrame:
    """Interval merge with score summing — numpy-based, no iterrows."""
    out_chrs, out_starts, out_ends, out_scores = [], [], [], []
    for chrom, grp in bg.groupby("chr", sort=False):
        starts = grp["start"].to_numpy(dtype=np.int64)
        ends   = grp["end"].to_numpy(dtype=np.int64)
        scores = grp["score"].to_numpy(dtype=np.float64)
        order  = np.argsort(starts, kind="stable")
        starts, ends, scores = starts[order], ends[order], scores[order]

        ms, me, msc = starts[0], ends[0], scores[0]
        for s, e, sc in zip(starts[1:], ends[1:], scores[1:]):
            if s <= me:
                me  = max(me, e)
                msc += sc
            else:
                out_chrs.append(chrom); out_starts.append(ms)
                out_ends.append(me);   out_scores.append(msc)
                ms, me, msc = s, e, sc
        out_chrs.append(chrom); out_starts.append(ms)
        out_ends.append(me);   out_scores.append(msc)

    return pd.DataFrame({
        "chr": out_chrs, "start": out_starts,
        "end": out_ends, "score": out_scores,
    })


def _annotate_with_bedtools(bg: pd.DataFrame, bim: pd.DataFrame) -> np.ndarray:
    """Annotate BIM SNPs using bedtools intersect."""
    bim_bed = bim.copy()
    bim_bed["_start"] = bim_bed["BP"] - 1  # 0-based
    bim_bed["_end"] = bim_bed["BP"]
    bim_bed["_chr"] = bim_bed["CHR"].astype(str)

    import tempfile as _tf
    fa = _tf.NamedTemporaryFile(mode="w", suffix=".bed", delete=False)
    bim_out = pd.DataFrame({
        "chr":   bim_bed["_chr"],
        "start": bim_bed["_start"],
        "end":   bim_bed["_end"],
        "idx":   bim_bed.index,
    })
    bim_out.to_csv(fa, sep="\t", header=False, index=False)
    fa.close()
    snp_bed_file = fa.name

    fb = _tf.NamedTemporaryFile(mode="w", suffix=".bedgraph", delete=False)
    bg_out = bg.copy()
    bg_out["chr"] = bg_out["chr"].astype(str).str.replace("chr", "", regex=False)
    bg_out[["chr", "start", "end", "score"]].to_csv(fb, sep="\t", header=False, index=False)
    fb.close()
    bg_file = fb.name

    try:
        result = subprocess.run(
            [
                "bedtools", "intersect",
                "-a", snp_bed_file,
                "-b", bg_file,
                "-wa", "-wb",
            ],
            capture_output=True, text=True, check=True,
        )
        annot_values = np.zeros(len(bim))
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            snp_idx = int(parts[3])
            score = float(parts[7])
            annot_values[snp_idx] += score
        return annot_values
    except subprocess.CalledProcessError:
        logger.warning("bedtools intersect failed; falling back to Python annotation")
        return _annotate_python(bg, bim)
    finally:
        for f in [snp_bed_file, bg_file]:
            try:
                os.unlink(f)
            except Exception:
                pass


def _annotate_python(bg: pd.DataFrame, bim: pd.DataFrame) -> np.ndarray:
    """Pure-Python SNP annotation from bedgraph."""
    bg = bg.copy()
    bg["_chr"] = bg["chr"].astype(str).str.replace("^chr", "", regex=True)
    bim = bim.copy()
    bim["_chr"] = bim["CHR"].astype(str)

    annot_values = np.zeros(len(bim))

    for chrom, bg_chrom in bg.groupby("_chr"):
        snp_mask = bim["_chr"] == str(chrom)
        if not snp_mask.any():
            continue
        snp_indices = np.where(snp_mask)[0]
        snp_pos = bim.loc[snp_mask, "BP"].values
        starts = bg_chrom["start"].values
        ends = bg_chrom["end"].values
        scores = bg_chrom["score"].values

        for idx, pos in zip(snp_indices, snp_pos):
            hits = np.where((starts <= pos) & (pos < ends))[0]
            if hits.size > 0:
                annot_values[idx] = scores[hits].sum()

    return annot_values


def _write_zero_annotation(bim_file: Union[str, Path], out_prefix: str) -> Path:
    """Write an all-zero annotation file for a chromosome with no coverage."""
    bim = pd.read_csv(
        bim_file, sep=r"\s+", header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
    )
    out_path = Path(f"{out_prefix}.annot.gz")
    annot = pd.DataFrame({"ANNOT": np.zeros(len(bim), dtype=np.float64)})
    annot.to_csv(out_path, sep="\t", index=False, compression="gzip")
    return out_path


def _parse_ldsc_results_file(results_file: Path) -> Optional[dict]:
    """
    Parse an LDSC ``.results`` file written by ``--h2 --overlap-annot``.

    The file is tab-separated with one row per annotation category.  The
    **last row** is always the program-specific annotation (``L2_1``).

    Columns used:
        ``Enrichment``, ``Enrichment_std_error``, ``Coefficient``,
        ``Coefficient_std_error``, ``Coefficient_z-score``.
    """
    if not results_file.exists():
        return None
    try:
        df = pd.read_csv(results_file, sep="\t")
    except Exception:
        return None

    if df.empty:
        return None

    row = df.iloc[-1]

    result: dict = {}
    for src_col, dst_col in [
        ("Enrichment",              "Enrichment"),
        ("Enrichment_std_error",    "Enrichment_std_error"),
        ("Coefficient",             "Coefficient"),
        ("Coefficient_std_error",   "Coefficient_std_error"),
        ("Coefficient_z-score",     "Coefficient_z_score"),
    ]:
        if src_col in df.columns:
            try:
                result[dst_col] = float(row[src_col])
            except (ValueError, TypeError):
                pass

    if "Enrichment" in result and "Enrichment_std_error" in result:
        e  = result["Enrichment"]
        se = result["Enrichment_std_error"]
        result["Enrichment_z_score"] = e / (se + 1e-12)

    return result if result else None


def _parse_ldsc_log(log_file: Path) -> Optional[dict]:
    """
    Parse an LDSC ``.log`` file and its companion ``.results`` file.

    LDSC writes enrichment/coefficient statistics to ``<prefix>.results``
    (tab-separated, one row per annotation) when ``--overlap-annot`` is used,
    and runtime metadata to ``<prefix>.log``.  This function reads both.

    Always extracted from ``.log`` (if present):
        ``h2_obs``, ``h2_obs_se`` — observed-scale heritability.

    Extracted from ``.results`` (when ``--overlap-annot`` succeeded):
        ``Enrichment``, ``Enrichment_std_error``, ``Enrichment_z_score``,
        ``Coefficient``, ``Coefficient_std_error``, ``Coefficient_z_score``.

    Returns ``None`` only if the file is missing or entirely unparseable.
    """
    if not log_file.exists():
        return None

    import re as _re
    text = log_file.read_text()

    result: dict = {}

    h2_match = _re.search(
        r"Total Observed scale h2:\s*([\-\d.eE+]+)\s*\(([\d.eE+]+)\)", text
    )
    if h2_match:
        result["h2_obs"]    = float(h2_match.group(1))
        result["h2_obs_se"] = float(h2_match.group(2))

    results_file = log_file.with_suffix(".results")
    stats = _parse_ldsc_results_file(results_file)
    if stats:
        result.update(stats)

    if not result:
        logger.warning(f"Could not parse any statistics from {log_file}")
        return None

    return result


def _is_float(s: str) -> bool:
    """Return True if *s* can be converted to a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _safe_filename(s: str) -> str:
    """Convert a string to a filesystem-safe filename (spaces → underscores, special chars stripped)."""
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"[^\w.\-+]+", "_", s)