import gzip
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

logger = logging.getLogger(__name__)


def _safe_name(s: str) -> str:
    """Convert a cell type name to a safe file name."""
    s = str(s).strip().replace(" ", "_")
    s = s.replace("(", "").replace(")", "")
    return re.sub(r"[^\w\.\+\-]+", "_", s)


def _chunk_rows(adata: AnnData, start: int, stop: int, layer: str | None) -> sparse.csr_matrix:
    """Read a block of rows as CSR (works for in-memory and backed AnnData)."""
    X = adata.layers[layer][start:stop] if layer is not None else adata.X[start:stop]
    return X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)


def pseudobulk_peak_counts(
    adata: AnnData,
    *,
    celltype_col: str,
    donor_col: str,
    obs_mask: np.ndarray | None = None,
    layer: str | None = None,
    chunk_size: int = 5000,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sum peak counts per (cell type, donor) pseudobulk.

    Reads the count matrix in row chunks, so backed AnnData (``backed="r"``) works
    without loading the full peak matrix into memory.

    Parameters
    ----------
    adata
        AnnData of cells x peaks with raw fragment or insertion counts.
    celltype_col
        Column in ``adata.obs`` with the cell type (or cell state) labels.
    donor_col
        Column in ``adata.obs`` with the donor labels.
    obs_mask
        Optional boolean mask over cells; cells outside the mask are ignored
        (e.g. one disease grade or one condition).
    layer
        Layer with the counts. Uses ``adata.X`` if None.
    chunk_size
        Number of cells read per chunk.

    Returns
    -------
    counts
        Pseudobulks x peaks DataFrame of summed counts, indexed by a
        ``MultiIndex`` (cell type, donor).
    n_cells
        Number of cells per pseudobulk, aligned to ``counts``.
    """
    obs = adata.obs
    ct = obs[celltype_col].astype(str).values
    dn = obs[donor_col].astype(str).values
    keep = np.ones(adata.n_obs, bool) if obs_mask is None else np.array(obs_mask, dtype=bool)
    keep &= pd.notna(obs[celltype_col]).values & pd.notna(obs[donor_col]).values
    key = pd.MultiIndex.from_arrays([ct, dn], names=["cell_type", "donor"])
    index = pd.MultiIndex.from_tuples(sorted(set(key[keep])), names=["cell_type", "donor"])
    codes = np.where(keep, index.get_indexer(key), -1)
    sums = np.zeros((len(index), adata.n_vars), np.float64)
    for s0 in range(0, adata.n_obs, chunk_size):
        e = min(s0 + chunk_size, adata.n_obs)
        c = codes[s0:e]
        m = c >= 0
        if not m.any():
            continue
        G = sparse.csr_matrix((np.ones(m.sum()), (c[m], np.flatnonzero(m))), shape=(len(index), e - s0))
        sums += (G @ _chunk_rows(adata, s0, e, layer)).toarray()
    n_cells = pd.Series(np.bincount(codes[codes >= 0], minlength=len(index)), index=index)
    return pd.DataFrame(sums, index=index, columns=adata.var_names), n_cells


def preprocess_for_chromatin_sldsc(
    adata: AnnData,
    *,
    celltype_col: str,
    donor_col: str,
    obs_mask: np.ndarray | None = None,
    min_cells: int = 20,
    min_donors: int = 1,
    layer: str | None = None,
    chunk_size: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Compute donor-aware peak accessibility and specificity per cell type for S-LDSC.

    The chromatin counterpart of :func:`preprocess_for_sldsc`:

    1. pseudobulk counts per (cell type, donor), keeping pseudobulks with at least ``min_cells`` cells;
    2. counts per million per pseudobulk;
    3. mean CPM over donors per cell type (each donor weighted equally);
    4. specificity = mean CPM of the cell type / sum of the mean CPM over all cell types (per peak).

    Cell types with fewer than ``min_donors`` donors are dropped.

    Parameters
    ----------
    adata
        AnnData of cells x peaks with raw counts (in memory or backed).
    celltype_col
        Column in ``adata.obs`` with the cell type labels.
    donor_col
        Column in ``adata.obs`` with the donor labels.
    obs_mask
        Optional boolean mask over cells (e.g. one disease grade).
    min_cells
        Minimum number of cells for a (cell type, donor) pseudobulk to be used.
    min_donors
        Minimum number of donors for a cell type to be kept.
    layer
        Layer with the counts. Uses ``adata.X`` if None.
    chunk_size
        Number of cells read per chunk.

    Returns
    -------
    mean_cpm
        Peaks x cell types DataFrame of donor-averaged CPM.
    specificity
        Peaks x cell types DataFrame of specificity scores (rows sum to 1 where accessible).
    n_donors
        Number of donors with a usable pseudobulk (≥ ``min_cells`` cells) per cell type,
        including cell types dropped by ``min_donors``.

    Examples
    --------
    >>> mean_cpm, specificity, n_donors = preprocess_for_chromatin_sldsc(
    ...     adata_peaks, celltype_col="cell_state", donor_col="donor", min_cells=20, min_donors=3
    ... )
    """
    counts, n_cells = pseudobulk_peak_counts(
        adata, celltype_col=celltype_col, donor_col=donor_col, obs_mask=obs_mask, layer=layer, chunk_size=chunk_size
    )
    counts = counts[n_cells.values >= min_cells]
    libsize = counts.sum(axis=1).replace(0, np.nan)
    cpm = counts.div(libsize, axis=0) * 1e6
    n_donors = cpm.groupby(level="cell_type").size()
    mean_cpm = cpm.groupby(level="cell_type").mean().T
    keep = n_donors.index[n_donors >= min_donors]
    dropped = sorted(set(n_donors.index) - set(keep))
    if dropped:
        logger.warning(f"Dropping cell types with < {min_donors} donors: {dropped}")
    mean_cpm = mean_cpm[list(keep)]
    mean_cpm.index.name = "peak"
    total = mean_cpm.sum(axis=1).replace(0, np.nan)
    specificity = mean_cpm.div(total, axis=0).fillna(0.0)
    logger.info(f"Peak specificity for {mean_cpm.shape[1]} cell types over {mean_cpm.shape[0]} peaks")
    return mean_cpm, specificity, n_donors


def generate_sldsc_peaksets(
    specificity: pd.DataFrame,
    peaks: pd.DataFrame,
    *,
    out_dir: str | Path,
    top_frac: float = 0.10,
    n_top: int | None = None,
    candidates: pd.DataFrame | None = None,
    universe: pd.Series | np.ndarray | None = None,
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
    include_control: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Write cell-type-specific peak sets as BED files for S-LDSC.

    For each cell type, the ``n_top`` peaks with the highest specificity among that
    cell type's candidate peaks are selected (stable sort, so ties keep peak order).
    Peaks without coordinates in ``peaks`` (e.g. failed liftover to the reference
    panel's build) are selected as usual but not written. A ``control.bed`` with
    every universe peak that has coordinates is written as the all-peaks control
    annotation (LDSC-SEG design; Finucane et al. 2018).

    Parameters
    ----------
    specificity
        Peaks x cell types specificity (e.g. from :func:`preprocess_for_chromatin_sldsc`).
    peaks
        Peak coordinates indexed like ``specificity`` (same build as the reference
        panel, BED convention: 0-based start, end exclusive). Missing coordinates
        mark peaks that cannot be annotated.
    out_dir
        Output directory for ``<cell_type>.bed`` and ``control.bed``.
    top_frac
        Fraction of the universe selected per cell type (used if ``n_top`` is None).
    n_top
        Number of peaks selected per cell type. Defaults to ``round(top_frac * universe size)``.
    candidates
        Optional peaks x cell types boolean DataFrame of peaks eligible per cell type
        (e.g. peaks called in that cell type, or accessible above a threshold).
    universe
        Optional boolean mask over peaks (e.g. outside the ENCODE blacklist). Only
        universe peaks are selected or written to the control.
    chrom_col, start_col, end_col
        Coordinate columns in ``peaks``.
    include_control
        Whether to write ``control.bed``.
    overwrite
        Whether to write into an existing directory.

    Returns
    -------
    pd.DataFrame
        One row per cell type (and the control): number of candidates, selected
        and written peaks, and the BED file path.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory {out_dir} is not empty. Set overwrite=True to proceed.")
    out_dir.mkdir(parents=True, exist_ok=True)
    peaks = peaks.reindex(specificity.index)
    if universe is None:
        uni = np.ones(len(peaks), bool)
    elif isinstance(universe, pd.Series):
        uni = universe.reindex(peaks.index).fillna(False).values.astype(bool)
    else:
        uni = np.asarray(universe, bool)
    has_coord = peaks[[chrom_col, start_col, end_col]].notna().all(axis=1).values
    n_top = int(round(top_frac * uni.sum())) if n_top is None else int(n_top)
    rows = []

    def _write(path: Path, sel: np.ndarray) -> int:
        bed = peaks.loc[sel & has_coord & uni, [chrom_col, start_col, end_col]].copy()
        bed[start_col], bed[end_col] = bed[start_col].astype(np.int64), bed[end_col].astype(np.int64)
        bed.to_csv(path, sep="\t", header=False, index=False)
        return len(bed)

    for ct in specificity.columns:
        cand = uni.copy()
        if candidates is not None:
            cand &= candidates[ct].reindex(peaks.index).fillna(False).values.astype(bool)
        idx = np.flatnonzero(cand)
        top = idx[np.argsort(-specificity[ct].values[idx], kind="stable")[:n_top]]
        sel = np.zeros(len(peaks), bool)
        sel[top] = True
        path = out_dir / f"{_safe_name(ct)}.bed"
        n_sel, n_written = int(sel.sum()), _write(path, sel)
        rows.append(dict(cell_type=ct, candidates=len(idx), selected=n_sel, written=n_written, bed_file=str(path)))
    if include_control:
        path = out_dir / "control.bed"
        n_uni = int(uni.sum())
        rows.append(
            dict(cell_type="control", candidates=n_uni, selected=n_uni, written=_write(path, uni), bed_file=str(path))
        )
    summary = pd.DataFrame(rows)
    logger.info(f"Wrote {len(summary)} peak sets ({n_top} peaks per cell type) to {out_dir}")
    return summary


def make_annot_from_bed_regions(
    bimfile: str,
    annot_file: str,
    bed_files: dict[str, str],
    bed_genome_build: str | None = None,
    bim_genome_build: str | None = None,
) -> dict:
    """
    Create a binary S-LDSC annotation from BED regions without bedtools.

    A SNP is annotated with 1 if its position lies in any region of the BED file
    (BED convention: 0-based start, end exclusive, i.e. ``start <= BP - 1 < end``),
    after merging overlapping regions. This is the rule of LDSC's
    ``make_annot.py --bed-file``. Several BED files give a multi-column annotation,
    one column per key of ``bed_files``, for computing LD scores of many
    annotations in one ``ldsc.py --l2`` call (see :func:`split_ld_scores`).

    Unlike ``make_annot_from_bimfile(bed_file=...)``, which calls ``make_annot.py``
    (and needs pybedtools in the LDSC environment), this runs in Python, takes several
    BED files at once and writes them as columns of one annotation file.

    Parameters
    ----------
    bimfile
        PLINK .bim file of one chromosome of the reference panel.
    annot_file
        Output annotation file (``.annot.gz``), columns CHR, BP, SNP, CM and one per BED file.
    bed_files
        Mapping from annotation name to a BED file in the reference panel's build.
    bed_genome_build, bim_genome_build
        Genome builds of the BED files and of ``bimfile`` (e.g. ``"GRCh37"``). If both are
        given and differ, raises; peaks called on GRCh38 must be lifted to the panel's
        build first.

    Returns
    -------
    dict
        ``annot_file``, ``files_created`` and ``n_nonzero_snps`` (annotated SNPs per annotation).
    """
    if bed_genome_build is not None and bim_genome_build is not None and bed_genome_build != bim_genome_build:
        raise ValueError(
            f"Genome build mismatch: BED files are {bed_genome_build!r} but bimfile is {bim_genome_build!r}. "
            "Lift the regions over to the reference panel's build first."
        )
    if bed_genome_build is None or bim_genome_build is None:
        logger.warning(
            "make_annot_from_bed_regions: bed_genome_build/bim_genome_build not both given; no genome-build check done."
        )
    bim = pd.read_csv(
        bimfile, sep=r"\s+", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"], dtype={"SNP": str}
    )
    chrom = str(bim.CHR.iloc[0]).removeprefix("chr")
    pos0 = bim.BP.to_numpy(np.int64) - 1
    out = bim[["CHR", "BP", "SNP", "CM"]].copy()
    n_annot = {}
    for name, bed_file in bed_files.items():
        bed = pd.read_csv(
            bed_file, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], dtype={"chrom": str}
        )
        bed = bed[bed.chrom.str.removeprefix("chr") == chrom].sort_values("start")
        inside = np.zeros(len(pos0), bool)
        if len(bed):
            s, e = bed.start.to_numpy(np.int64), bed.end.to_numpy(np.int64)
            ks, ke = [s[0]], [e[0]]
            for a, z in zip(s[1:], e[1:]):
                if a <= ke[-1]:
                    ke[-1] = max(ke[-1], z)
                else:
                    ks.append(a)
                    ke.append(z)
            ks, ke = np.asarray(ks), np.asarray(ke)
            j = np.searchsorted(ks, pos0, side="right") - 1
            inside = (j >= 0) & (pos0 < np.where(j >= 0, ke[np.maximum(j, 0)], -1))
        out[name] = inside.astype(np.int8)
        n_annot[name] = int(inside.sum())
    os.makedirs(os.path.dirname(os.path.abspath(annot_file)), exist_ok=True)
    with gzip.open(annot_file, "wt") if annot_file.endswith(".gz") else open(annot_file, "w") as fh:
        out.to_csv(fh, sep="\t", index=False)
    logger.info(f"Wrote {annot_file}: " + ", ".join(f"{k} {v:,} SNPs" for k, v in n_annot.items()))
    return {"annot_file": annot_file, "files_created": [annot_file], "n_nonzero_snps": n_annot}


def split_ld_scores(ld_prefix: str, out_prefixes: dict[str, str]) -> dict[str, str]:
    """
    Split a multi-annotation LD score file into one single-annotation file per column.

    ``ldsc.py --l2`` computes each annotation column independently, so LD scores of
    many annotations can be computed in one call and split afterwards into the
    per-annotation files that ``ldsc.py --h2-cts`` expects (one prefix per line of the
    ``--ref-ld-chr-cts`` file).

    Parameters
    ----------
    ld_prefix
        Prefix of one chromosome's multi-annotation output, e.g. ``ld_all.22`` for
        ``ld_all.22.l2.ldscore.gz``, ``.l2.M`` and ``.l2.M_5_50``.
    out_prefixes
        Mapping from annotation column name (as in the ``.annot.gz`` header) to the
        output prefix for that chromosome, e.g. ``{"Hypertrophic": "cts/ld_Hypertrophic.22"}``.

    Returns
    -------
    dict
        Mapping from annotation name to the written ``.l2.ldscore.gz`` file.
    """
    ld = pd.read_csv(f"{ld_prefix}.l2.ldscore.gz", sep="\t")
    cols = [c for c in ld.columns if c not in ("CHR", "SNP", "BP")]
    m = {ext: Path(f"{ld_prefix}.{ext}").read_text().split() for ext in ("l2.M", "l2.M_5_50")}
    written = {}
    for name, out in out_prefixes.items():
        i = cols.index(f"{name}L2") if f"{name}L2" in cols else cols.index(name)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        ld[["CHR", "SNP", "BP", cols[i]]].rename(columns={cols[i]: "L2"}).to_csv(
            f"{out}.l2.ldscore.gz", sep="\t", index=False, compression="gzip"
        )
        for ext, vals in m.items():
            Path(f"{out}.{ext}").write_text(vals[i] + "\n")
        written[name] = f"{out}.l2.ldscore.gz"
    return written


def write_cts_file(ld_prefixes: dict[str, str], cts_file: str, control_prefix: str | None = None) -> str:
    """
    Write a ``--ref-ld-chr-cts`` file for :func:`estimate_celltype_specific_heritability`.

    Each line is ``<name>\\t<prefix>[,<control prefix>]``. With a control (e.g. all
    peaks, or all genes), each cell type annotation is tested conditional on the
    control and the baseline model (LDSC-SEG; Finucane et al. 2018).

    Parameters
    ----------
    ld_prefixes
        Mapping from cell type name to its LD score prefix without the chromosome
        number (e.g. ``"cts/ld_Hypertrophic."``).
    cts_file
        Output path.
    control_prefix
        Optional LD score prefix of the control annotation.

    Returns
    -------
    str
        The path of the written file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(cts_file)), exist_ok=True)
    with open(cts_file, "w") as fh:
        for name, prefix in ld_prefixes.items():
            fh.write(f"{_safe_name(name)}\t{prefix}" + (f",{control_prefix}" if control_prefix else "") + "\n")
    return cts_file
