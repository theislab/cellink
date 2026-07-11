import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.nn.conv import MessagePassing

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _AggNet(MessagePassing):
        """
        Single graph convolution layer from model/model.py.

        Weighted mean aggregation with two learnable scalar parameters.

        Code adapted from Zhang et al. (2025): https://doi.org/10.1038/s41587-025-02725-6 / https://github.com/szhang1112/scPRS
        """

        def __init__(self, **kwargs):
            kwargs.setdefault("aggr", "mean")
            super().__init__(**kwargs)
            self.parameter1 = nn.Parameter(torch.FloatTensor(abs(np.random.rand(1)).reshape(1, 1)))
            self.parameter2 = nn.Parameter(torch.FloatTensor(abs(np.random.rand(1)).reshape(1, 1)))
            self.bias = nn.Parameter(torch.FloatTensor(np.zeros(1).reshape(1, 1)))

        def forward(self, x, edge_index, edge_weight):
            out = self.propagate(edge_index=edge_index, edge_weight=edge_weight, x=x)
            out += self.bias
            return F.leaky_relu(out, negative_slope=0.1)

        def message(self, x_i, x_j, edge_weight):
            return abs(self.parameter1) * x_j + abs(self.parameter2) * x_i

    class _scPRSModel(nn.Module):
        """
        scPRS GNN from model/model.py.

        Code adapted from Zhang et al. (2025): https://doi.org/10.1038/s41587-025-02725-6 / https://github.com/szhang1112/scPRS

        Parameters
        ----------
        dim_in : int
            Number of PRS features per cell (21 in the paper: 7 p-thresholds x 3 r² values).
        n_cell : int
            Number of reference cells.
        n_gcn : int
            Number of graph convolution layers (default 3).
        """

        def __init__(self, dim_in: int, n_cell: int, n_gcn: int):
            super().__init__()
            self.parameter = nn.Parameter(torch.FloatTensor(abs(np.random.rand(dim_in).reshape([dim_in, 1]))))
            self.bias1 = nn.Parameter(torch.FloatTensor(np.zeros(1).reshape(1, 1)))
            self.agg = nn.ModuleList([_AggNet() for _ in range(n_gcn)])
            self.pred = nn.Linear(n_cell, 1)

        def forward(self, x, edge_index, edge_weight):
            # x: (batch, n_cell, dim_in)
            x = (x @ abs(self.parameter) / x.shape[-1] + self.bias1).transpose(0, 1).squeeze(-1)
            for net in self.agg:
                x = net(x=x, edge_index=edge_index, edge_weight=edge_weight)
            out = x.transpose(0, 1)
            out = self.pred(out)
            return out


def _normalise_gwas_df(gwas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise GWAS summary statistics to standard column names.

    Handles the GWAS Catalog harmonised format returned by
    ``get_gwas_catalog_study_summary_stats``:
    ``chromosome``, ``base_pair_location``, ``effect_allele``,
    ``other_allele``, ``p_value``, ``beta`` / ``odds_ratio``.

    When no SNP identifier column is present, a synthetic ``CHR:BP``
    identifier is created (e.g. ``1:752721``).
    """
    df = gwas_df.copy()

    rename = {
        "chromosome": "CHR",
        "base_pair_location": "BP",
        "effect_allele": "A1",
        "other_allele": "A2",
        "p_value": "P",
        "beta": "BETA",
        "odds_ratio": "OR",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "BETA" not in df.columns and "OR" in df.columns:
        df["BETA"] = np.log(df["OR"].astype(float))

    has_snp = any(c in df.columns for c in ["SNP", "rsid", "ID", "variant_id"])
    if not has_snp:
        if "CHR" in df.columns and "BP" in df.columns:
            logger.info("No SNP/rsid column — synthesising variant IDs as 'CHR:BP'.")
            df["SNP"] = (
                df["CHR"].astype(str).str.replace("chr", "", regex=False)
                + ":"
                + df["BP"].astype(float).astype(int).astype(str)
            )
        else:
            raise ValueError(
                "Cannot determine variant identifiers: no SNP/rsid column "
                "and no CHR+BP columns to synthesise one. "
                f"Available columns: {list(df.columns)}"
            )
    else:
        for c in ["rsid", "ID", "variant_id"]:
            if c in df.columns and "SNP" not in df.columns:
                df = df.rename(columns={c: "SNP"})
                break

    return df


def _ids_look_like_rsids(ids: pd.Series, sample: int = 200) -> bool:
    """Return True if the majority of sampled IDs start with 'rs'."""
    s = ids.dropna().head(sample).astype(str)
    return (s.str.startswith("rs")).mean() > 0.5


def _annotate_gwas_with_bim_ids(
    gwas_df: pd.DataFrame,
    bfile: str | Path,
) -> pd.DataFrame:
    """
    Harmonise SNP IDs between GWAS summary stats and a PLINK bfile.

    This lets PLINK ``--clump`` and ``--score`` find variants by ID.
    Position-based replacement requires the GWAS and bfile to share the same
    genome build.  If the match rate is below 20 % a
    warning is emitted because this usually indicates a build mismatch rather
    than a genuine absence of GWAS variants in the reference panel.
    """
    bim_path = str(bfile) + ".bim"
    if not os.path.exists(bim_path):
        return gwas_df

    bim = pd.read_csv(
        bim_path,
        sep="\t",
        header=None,
        names=["CHR", "SNP", "cM", "BP", "A1", "A2"],
    )

    gwas_has_rsids = _ids_look_like_rsids(gwas_df["SNP"])
    bim_has_rsids = _ids_look_like_rsids(bim["SNP"])

    if gwas_has_rsids and bim_has_rsids:
        logger.info(
            "SNP ID annotation: GWAS and bim both use rsIDs — skipping "
            "CHR:BP lookup (direct rsID matching will be used by PLINK)."
        )
        return gwas_df

    bim["_key"] = bim["CHR"].astype(str).str.replace("chr", "", regex=False) + ":" + bim["BP"].astype(int).astype(str)
    bim_map: dict[str, str] = bim.drop_duplicates("_key").set_index("_key")["SNP"].to_dict()

    gwas_key = (
        gwas_df["CHR"].astype(str).str.replace("chr", "", regex=False)
        + ":"
        + gwas_df["BP"].astype(float).astype(int).astype(str)
    )
    matched = gwas_key.map(bim_map)
    n_matched = int(matched.notna().sum())
    match_pct = 100.0 * n_matched / max(len(gwas_df), 1)
    target_fmt = "CHR:BP" if not bim_has_rsids else "rsIDs"
    logger.info(
        f"SNP ID annotation: matched {n_matched}/{len(gwas_df)} ({match_pct:.1f}%) "
        f"GWAS variants to bim {target_fmt} via CHR:BP position lookup."
    )
    if match_pct < 20.0:
        logger.warning(
            f"Only {match_pct:.1f}% of GWAS variants matched by position — "
            "this typically means a genome-build mismatch between the GWAS "
            "(e.g. GRCh38) and the bfile (e.g. GRCh37).  "
            "Fetch the GWAS in the same build as your bfile, or supply a "
            "matching reference panel."
        )
    if n_matched > 0:
        gwas_df = gwas_df.copy()
        gwas_df["SNP"] = matched.where(matched.notna(), gwas_df["SNP"])
    return gwas_df


def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not find any of {candidates} in columns: {list(df.columns)}")
    return None


def _build_peak_snp_map(
    adata: AnnData,
    gwas_df: pd.DataFrame,
    snp_col: str,
) -> list[list[str]]:
    """
    Build a peak→SNP map across ALL peaks in ``adata.var`` in one pass.

    Returns a list of length ``adata.n_vars`` where ``result[i]`` is the
    (possibly empty) list of GWAS SNP IDs whose genomic position falls inside
    peak ``i``.  Computed once and shared across all per-cell queries so that
    per-cell work is reduced to ``np.nonzero(peak_mask)`` + list indexing.

    Uses NCLS for O(n_snps log n_peaks) bulk overlap if available; falls back
    to a sorted-peaks + searchsorted approach with no extra dependencies.
    """
    chr_col = _find_col(gwas_df, ["CHR", "chr", "chrom", "#CHROM"])
    bp_col = _find_col(gwas_df, ["BP", "bp", "POS", "pos"])

    var = adata.var.copy()
    var["_chrom"] = var["chrom"].astype(str).str.replace("chr", "", regex=False)
    var["_start"] = var["start"].astype(int)
    var["_end"] = var["end"].astype(int)
    var["_vidx"] = np.arange(adata.n_vars)

    gwas = gwas_df[[snp_col, chr_col, bp_col]].copy()
    gwas["_chrom"] = gwas[chr_col].astype(str).str.replace("chr", "", regex=False)
    gwas["_bp"] = gwas[bp_col].astype(int)

    result: list[list[str]] = [[] for _ in range(adata.n_vars)]

    try:
        from ncls import NCLS

        for chrom, peak_grp in var.groupby("_chrom"):
            gwas_chrom = gwas[gwas["_chrom"] == chrom]
            if gwas_chrom.empty:
                continue
            p_starts = peak_grp["_start"].values.astype(np.int64)
            p_ends = (peak_grp["_end"].values + 1).astype(np.int64)  # half-open
            p_local = np.arange(len(peak_grp), dtype=np.int64)
            tree = NCLS(p_starts, p_ends, p_local)

            q_starts = gwas_chrom["_bp"].values.astype(np.int64)
            q_ends = (q_starts + 1).astype(np.int64)
            q_ids = np.arange(len(gwas_chrom), dtype=np.int64)
            snp_local_arr, peak_local_arr = tree.all_overlaps_both(q_starts, q_ends, q_ids)

            p_global = peak_grp["_vidx"].values
            snp_ids = gwas_chrom[snp_col].values
            for snp_local, pk_local in zip(snp_local_arr, peak_local_arr, strict=False):
                result[p_global[pk_local]].append(snp_ids[snp_local])

    except ImportError:
        for chrom, peak_grp in var.groupby("_chrom"):
            gwas_chrom = gwas[gwas["_chrom"] == chrom]
            if gwas_chrom.empty:
                continue
            p_starts = peak_grp["_start"].values
            p_ends = peak_grp["_end"].values
            p_global = peak_grp["_vidx"].values
            order = np.argsort(p_starts)
            s_starts = p_starts[order]
            s_ends = p_ends[order]
            s_global = p_global[order]
            bps = gwas_chrom["_bp"].values
            snp_ids = gwas_chrom[snp_col].values
            for bp, snp_id in zip(bps, snp_ids, strict=False):
                right = int(np.searchsorted(s_starts, bp, side="right"))
                if right == 0:
                    continue
                for k in np.where(s_ends[:right] >= bp)[0]:
                    result[s_global[k]].append(snp_id)

    return result


def _read_prs_score_files(
    prs_dir: str,
    r2_thresholds: list[float],
    p_thresholds: list[float],
    labels: pd.Series,
) -> tuple[np.ndarray, dict, dict, dict, dict]:
    """
    Adaptation of ``step1_process_prs_data`` from ``util/generate_data.py``.

    See https://github.com/szhang1112/scPRS/blob/main/util/generate_data.py.
    Reads PLINK score files written by the SLURM array job and assembles
    the ``(n_samples, n_cells, n_features)`` PRS matrix ``X``.

    The feature ordering matches ``step4_merge_data``:
        for p in ['0.1','0.05','0.01','0.5','1e-3','1e-4','1e-5']:
            for r2 in [0.1, 0.3, 0.5]:
                features.append(...)
    i.e. 7 x 3 = 21 features.
    """
    res = []
    for r2 in r2_thresholds:
        r2_dir = os.path.join(prs_dir, str(r2))
        if not os.path.isdir(r2_dir):
            continue
        for filename in os.listdir(r2_dir):
            if "profile" not in filename and "sscore" not in filename:
                continue
            filepath = os.path.join(r2_dir, filename)
            try:
                data = pd.read_csv(filepath, sep=r"\s+")
            except Exception:  # noqa: BLE001
                continue
            data["sim_cutoff"] = r2
            data["filename"] = filename
            res.append(data)

    if not res:
        raise RuntimeError(
            f"No PLINK score files found under {prs_dir}. " "Make sure the SLURM array job completed successfully."
        )

    res = pd.concat(res, ignore_index=True)

    res["cell"] = res["filename"].apply(lambda x: x.split(".")[0])
    res["type2"] = res["filename"].apply(lambda x: ".".join(x.split(".")[1:-1]))

    score_col = "SCORE1_AVG" if "SCORE1_AVG" in res.columns else "SCORE"

    d_type2 = {v: i for i, v in enumerate(sorted(set(res["type2"])))}
    d_sim_cutoff = {v: i for i, v in enumerate(sorted(set(res["sim_cutoff"])))}
    d_sample = {v: i for i, v in enumerate(sorted(set(res["IID"].astype(str))))}
    d_cell = {v: i for i, v in enumerate(sorted(set(res["cell"])))}

    X = np.zeros(
        [len(d_sample), len(d_cell), len(d_type2) * len(d_sim_cutoff)],
        dtype=np.float32,
    )

    for row in res.to_dict("records"):
        si = d_sample.get(str(row["IID"]))
        ci = d_cell.get(row["cell"])
        fi = d_sim_cutoff[row["sim_cutoff"]] * len(d_type2) + d_type2[row["type2"]]
        if si is not None and ci is not None:
            X[si, ci, fi] = row[score_col]

    p_order = [str(p) for p in p_thresholds]
    r2_order = sorted(r2_thresholds)
    feature_idx = []
    for p in p_order:
        for r2 in r2_order:
            fi = d_sim_cutoff.get(r2, 0) * len(d_type2) + d_type2.get(p, 0)
            feature_idx.append(fi)

    if len(feature_idx) == X.shape[2]:
        X = X[:, :, feature_idx]

    return X, d_sample, d_cell, d_type2, d_sim_cutoff


def _build_knn_graph(
    adata: AnnData,
    k: int = 25,
    n_pcs: int = 20,
    embed_key: str | None = None,
) -> np.ndarray:
    """
    Adaptation of ``step2_process_atac_data`` from ``util/generate_data.py``.

    See https://github.com/szhang1112/scPRS/blob/main/util/generate_data.py.
    Uses ``sklearn.neighbors.kneighbors_graph`` on the LSI/PCA embedding. Returns edge array of shape (n_edges, 2).
    """
    if embed_key is not None and embed_key in adata.obsm:
        atac_embed = adata.obsm[embed_key]
    elif "X_lsi" in adata.obsm:
        atac_embed = adata.obsm["X_lsi"]
    elif "X_pca" in adata.obsm:
        atac_embed = adata.obsm["X_pca"]
    else:
        logger.info("No LSI/PCA found — computing PCA.")
        import scipy.sparse as sp

        X_bin = (adata.X > 0).astype(float)
        if not sp.issparse(X_bin):
            X_bin = sp.csr_matrix(X_bin)
        adata_tmp = adata.copy()
        adata_tmp.X = X_bin
        sc.pp.normalize_total(adata_tmp, target_sum=1e4)
        sc.pp.log1p(adata_tmp)
        sc.pp.highly_variable_genes(adata_tmp, n_top_genes=min(2000, adata_tmp.n_vars))
        sc.pp.scale(adata_tmp, max_value=10)
        sc.tl.pca(adata_tmp, n_comps=min(50, adata_tmp.n_obs - 1))
        adata.obsm["X_pca"] = adata_tmp.obsm["X_pca"]
        atac_embed = adata.obsm["X_pca"]

    atac_embed = atac_embed[:, :n_pcs]

    logger.info(f"Building kNN graph (k={k}) from embedding of shape {atac_embed.shape}.")
    neighbor = kneighbors_graph(
        atac_embed,
        k,
        mode="distance",
        metric="euclidean",
        include_self=False,
    )
    neighbor = neighbor.toarray().T
    rows, cols = np.nonzero(neighbor)
    edge = np.column_stack([rows, cols]).astype(np.int64)
    logger.info(f"kNN graph: {len(edge)} edges over {adata.n_obs} cells.")
    return edge


def get_plink_commands_per_cell(
    adata: AnnData,
    gwas_file: str | Path,
    plink_bfile: str | Path,
    prs_dir: str | Path,
    ld_bfile: str | Path = None,
    p_thresholds: list[float] | None = None,
    r2_thresholds: list[float] | None = None,
    clump_kb: int = 250,
    maf: float = 0.01,
    plink_cmd: str = "plink",
    plink_threads: int = 4,
) -> list[str]:
    """
    Generate the PLINK shell commands for all cells without running them.

    Pre-generates per-cell filtered GWAS files in ``prs_dir`` (requires the
    AnnData object), then returns one self-contained shell command per cell.
    Use this to submit per-cell PLINK jobs as a cluster array job instead of
    running them in-process. For SLURM submission see
    :func:`write_slurm_array_job`.

    Parameters
    ----------
    adata : AnnData
        Reference scATAC-seq AnnData with peak coordinates in ``adata.var``.
    gwas_file : str or Path
        Post-QC GWAS summary statistics.
    plink_bfile : str or Path
        PLINK binary prefix for scoring.
    prs_dir : str or Path
        Directory where per-cell GWAS files and score files will be written.
    ld_bfile : str or Path, optional
        PLINK binary prefix for LD estimation. Defaults to ``plink_bfile``.
    p_thresholds : list of float, optional
        C+T p-value thresholds.
    r2_thresholds : list of float, optional
        C+T LD r² thresholds.
    clump_kb : int, default=250
        Clumping window in kb.
    maf : float, default=0.01
        MAF filter for scoring.
    plink_cmd : str, default="plink"
        PLINK executable name.
    plink_threads : int, default=4
        ``--threads`` value passed to every PLINK call.  Match this to the
        CPUs-per-task value when submitting to a cluster.

    Returns
    -------
    list of str
        One shell command string per cell (cells with no peak-overlapping
        GWAS variants are skipped).
    """
    p_thresholds = p_thresholds or [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5]
    r2_thresholds = r2_thresholds or [0.1, 0.3, 0.5]
    ld_bfile = ld_bfile or plink_bfile
    prs_dir = str(prs_dir)
    os.makedirs(prs_dir, exist_ok=True)

    gwas_df = pd.read_csv(gwas_file, sep="\t")
    gwas_df = _normalise_gwas_df(gwas_df)
    gwas_df = _annotate_gwas_with_bim_ids(gwas_df, plink_bfile)

    snp_col = "SNP"
    p_col = _find_col(gwas_df, ["P", "pval", "PVAL", "P_BOLT_LMM"])
    a1_col = _find_col(gwas_df, ["A1", "a1"])
    beta_col = _find_col(gwas_df, ["BETA", "beta", "Effect"])

    # Multiple GWAS positions can map to the same rsID (multi-allelic sites, near-identical lifted coordinates).  PLINK's --q-score-range errors on duplicate SNP IDs, so keep only the most significant hit per rsID.
    n_before = len(gwas_df)
    gwas_df = gwas_df.sort_values(p_col).drop_duplicates(snp_col, keep="first")
    n_dropped = n_before - len(gwas_df)
    if n_dropped:
        logger.info(f"Dropped {n_dropped} duplicate SNP IDs (kept most-significant per rsID).")

    full_gwas_file = os.path.join(prs_dir, "gwas_full.txt")
    snp_pval_file = os.path.join(prs_dir, "snp_pvalue.txt")
    range_file = os.path.join(prs_dir, "range_list.txt")
    gwas_df.to_csv(full_gwas_file, sep="\t", index=False)
    gwas_df[[snp_col, p_col]].to_csv(snp_pval_file, sep=" ", index=False, header=False)
    with open(range_file, "w") as fh:
        for pt in p_thresholds:
            fh.write(f"{pt} 0 {pt}\n")

    cols = list(gwas_df.columns)
    snp_pos = cols.index(snp_col) + 1
    a1_pos = cols.index(a1_col) + 1
    beta_pos = cols.index(beta_col) + 1

    logger.info("Building peak-SNP overlap map (computed once for all cells)")
    peak_snp_map = _build_peak_snp_map(adata, gwas_df, snp_col)

    commands = []
    for ci in tqdm(range(adata.n_obs)):
        if issparse(adata.X):
            peak_mask = np.asarray(adata.X[ci] > 0).ravel()
        else:
            peak_mask = adata.X[ci] > 0

        open_peaks = np.nonzero(peak_mask)[0]
        peak_snps_set: set[str] = set()
        for pi in open_peaks:
            peak_snps_set.update(peak_snp_map[pi])
        peak_snps = list(peak_snps_set)
        if not peak_snps:
            continue

        cell_id = adata.obs_names[ci]

        # Write per-cell GWAS file (peak-overlapping SNPs only).
        cell_gwas_file = os.path.join(prs_dir, f"_cell{ci}_gwas.txt")
        gwas_df[gwas_df[snp_col].isin(set(peak_snps))].to_csv(cell_gwas_file, sep="\t", index=False)

        cell_cmds = []
        for r2 in r2_thresholds:
            out_dir = os.path.join(prs_dir, str(r2))
            os.makedirs(out_dir, exist_ok=True)
            cell_out = os.path.join(out_dir, cell_id)
            tmp_clump = os.path.join(prs_dir, f"_cell{ci}_r2{r2}_clump")
            valid_snps = os.path.join(prs_dir, f"_cell{ci}_r2{r2}.valid.snp")

            clump_cmd = (
                f"{plink_cmd} --bfile {ld_bfile} "
                f"--clump {cell_gwas_file} {full_gwas_file} "
                f"--clump-snp-field {snp_col} --clump-field {p_col} "
                f"--clump-p1 1 --clump-r2 {r2} --clump-kb {clump_kb} "
                f"--clump-index-first "
                f"--threads {plink_threads} "
                f"--out {tmp_clump} --silent"
            )
            score_cmd = (
                f"{plink_cmd} --bfile {plink_bfile} "
                f"--score {cell_gwas_file} {snp_pos} {a1_pos} {beta_pos} header "
                f"--q-score-range {range_file} {snp_pval_file} "
                f"--extract {valid_snps} --maf {maf} "
                f"--threads {plink_threads} "
                f"--out {cell_out} --silent"
            )
            cell_cmds.append(
                f"{clump_cmd} && "
                f"if [ -f {tmp_clump}.clumped ]; then "
                f"awk 'NR>1{{print $3}}' {tmp_clump}.clumped > {valid_snps} && "
                f"{score_cmd}; "
                f"fi"
            )

        if cell_cmds:
            commands.append(" && ".join(cell_cmds))

    logger.info(f"Generated {len(commands)} per-cell PLINK command strings.")
    return commands


def write_slurm_array_job(
    adata: AnnData,
    gwas_file: str | Path,
    plink_bfile: str | Path,
    prs_dir: str | Path,
    output_dir: str | Path = ".",
    job_name: str = "scprs_plink",
    ld_bfile: str | Path = None,
    p_thresholds: list[float] | None = None,
    r2_thresholds: list[float] | None = None,
    clump_kb: int = 250,
    maf: float = 0.01,
    plink_cmd: str = "plink",
    plink_threads: int = 4,
    slurm_partition: str | None = None,
    slurm_qos: str | None = None,
    slurm_mem: str = "8G",
    slurm_cpus: int = 4,
    slurm_time: str = "04:00:00",
    slurm_max_simultaneous: int = 200,
    extra_sbatch_lines: list[str] | None = None,
) -> dict:
    """
    Prepare per-cell GWAS files and write a SLURM array job for the PLINK C+T step.

    1. Pre-generates per-cell filtered GWAS files in ``prs_dir`` (Python step,
       requires AnnData, must be run once before cluster submission).
    2. Writes ``{output_dir}/{job_name}_commands.txt`` — one shell command per
       cell, where each command runs PLINK clumping + scoring for all r²
       thresholds for that cell.
    3. Writes ``{output_dir}/{job_name}_array.sh`` — a SLURM array job script
       where task ``$SLURM_ARRAY_TASK_ID`` executes the corresponding line of
       the commands file.
    4. Logs the ``sbatch`` command and the follow-up Python call to run after
       all jobs complete.

    After all array tasks finish, load the results with::

        pkl = prepare_scprs_data(
            adata=adata,
            labels=labels,
            gwas_file=gwas_file,
            prefix="my_prefix",
            prs_dir="{prs_dir}",
        )

    Parameters
    ----------
    adata : AnnData
        Reference scATAC-seq AnnData with peak coordinates in ``adata.var``.
    gwas_file : str or Path
        Post-QC GWAS summary statistics.
    plink_bfile : str or Path
        PLINK binary prefix for the target cohort (scoring).
    prs_dir : str or Path
        Directory where per-cell GWAS files and PLINK score files are written.
        Pass this same path to :func:`prepare_scprs_data` afterwards.
    output_dir : str or Path, default="."
        Where to write the commands file and SLURM script.
    job_name : str, default="scprs_plink"
        SLURM job name and file prefix.
    ld_bfile : str or Path, optional
        PLINK prefix for LD estimation. Defaults to ``plink_bfile``.
    p_thresholds : list of float, optional
        C+T p-value thresholds.
    r2_thresholds : list of float, optional
        C+T LD r² thresholds.
    clump_kb : int, default=250
        Clumping window in kb.
    maf : float, default=0.01
        MAF filter for PLINK scoring.
    plink_cmd : str, default="plink"
        PLINK executable name (``plink`` or ``plink2``).
    plink_threads : int, default=4
        ``--threads`` passed to every PLINK call.  Should equal
        ``slurm_cpus`` so the CPU allocation is fully utilised.
    slurm_partition : str, optional
        ``#SBATCH --partition`` value (``-p`` flag).  If ``None``, the line
        is omitted — useful on clusters where the default partition is fine or
        where ``-q``/``--qos`` selects the queue instead.
    slurm_qos : str, optional
        ``#SBATCH -q`` value.  If ``None``, the line is omitted.
    slurm_mem : str, default="8G"
        Memory per task.  Clumping on 1000G EUR (~8 M variants, 503 samples)
        needs ~4-6 GB.  Increase for larger cohorts (e.g. ``"32G"`` for UKB).
    slurm_cpus : int, default=4
        ``--cpus-per-task``.  Should match ``plink_threads``.
    slurm_time : str, default="04:00:00"
        Wall-clock time limit per task (``HH:MM:SS``). Increase threads conservatively.
    slurm_max_simultaneous : int, default=200
        Maximum number of concurrently running array tasks (the ``%N`` suffix
        on ``--array``).  Reduce to avoid overwhelming shared storage.
    extra_sbatch_lines : list of str, optional
        Additional raw ``#SBATCH`` lines inserted verbatim, e.g.
        ``["#SBATCH --nice=0", "#SBATCH --gres=gpu:0"]``.

    Returns
    -------
    dict with keys:

    ``commands_file``
        Absolute path to the commands text file.
    ``script_file``
        Absolute path to the SLURM array script.
    ``n_cells``
        Number of cells with at least one peak-overlapping GWAS variant
        (= number of array tasks).
    ``submit_cmd``
        The ``sbatch`` command string to run.
    ``next_step``
        Reminder string describing the Python call to make after all jobs
        finish.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prs_dir_abs = str(Path(prs_dir).resolve())
    commands_file = str(output_dir / f"{job_name}_commands.txt")
    script_file = str(output_dir / f"{job_name}_array.sh")
    logs_dir = str(output_dir / "logs")
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Pre-generating per-cell GWAS files in {prs_dir_abs} " f"({adata.n_obs} cells) ...")
    adata.X = adata.X.todense()
    commands = get_plink_commands_per_cell(
        adata=adata,
        gwas_file=gwas_file,
        plink_bfile=str(plink_bfile),
        prs_dir=prs_dir_abs,
        ld_bfile=str(ld_bfile) if ld_bfile else None,
        p_thresholds=p_thresholds,
        r2_thresholds=r2_thresholds,
        clump_kb=clump_kb,
        maf=maf,
        plink_cmd=plink_cmd,
        plink_threads=plink_threads,
    )

    n_cells = len(commands)
    if n_cells == 0:
        raise RuntimeError(
            "No cells have peak-overlapping GWAS variants. "
            "Check that adata.var has 'chrom'/'start'/'end' columns and "
            "that the GWAS coordinates match the peak genome build."
        )

    with open(commands_file, "w") as fh:
        for cmd in commands:
            fh.write(cmd + "\n")
    logger.info(f"Wrote {n_cells} commands to {commands_file}")

    sbatch_lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{n_cells - 1}%{slurm_max_simultaneous}",
        f"#SBATCH --output={logs_dir}/{job_name}_%A_%a.out",
        f"#SBATCH --error={logs_dir}/{job_name}_%A_%a.err",
        f"#SBATCH --cpus-per-task={slurm_cpus}",
        f"#SBATCH --mem={slurm_mem}",
        f"#SBATCH --time={slurm_time}",
    ]
    if slurm_partition:
        sbatch_lines.append(f"#SBATCH -p {slurm_partition}")
    if slurm_qos:
        sbatch_lines.append(f"#SBATCH -q {slurm_qos}")
    if extra_sbatch_lines:
        sbatch_lines.extend(extra_sbatch_lines)

    script_body = "\n".join(
        [
            "#!/bin/bash",
            "# Generated by cellink.tl.external.write_slurm_array_job",
            "# ── SLURM directives ── edit to match your cluster ──────────────────",
            *sbatch_lines,
            "",
            "# ── runtime ──────────────────────────────────────────────────────────",
            "set -euo pipefail",
            "",
            f'COMMANDS_FILE="{commands_file}"',
            "",
            "# Read the command for this array task (tasks are 0-indexed, sed is 1-indexed)",
            'CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMMANDS_FILE")',
            'if [ -z "$CMD" ]; then',
            '    echo "Task $SLURM_ARRAY_TASK_ID: no command (cell skipped — no peak SNPs)."; exit 0',
            "fi",
            "",
            'echo "Task $SLURM_ARRAY_TASK_ID starting: $CMD"',
            'eval "$CMD"',
            'echo "Task $SLURM_ARRAY_TASK_ID done."',
        ]
    )

    with open(script_file, "w") as fh:
        fh.write(script_body + "\n")
    logger.info(f"Wrote SLURM array script to {script_file}")

    submit_cmd = f"sbatch {script_file}"
    next_step = (
        f"prepare_scprs_data("
        f"adata=adata, labels=labels, gwas_file=gwas_file, "
        f'prefix="<prefix>", prs_dir="{prs_dir_abs}")'
    )

    logger.info(
        f"\n"
        f"  Array job ready: {n_cells} tasks\n"
        f"  Submit with:     {submit_cmd}\n"
        f"\n"
        f"  Memory guidance:\n"
        f"    Current default: {slurm_mem} — suitable for 1000G EUR (~500 samples, ~8M variants).\n"
        f"    For UK Biobank scale (~400k samples): increase to 64G or more.\n"
        f"\n"
        f"  Timing guidance:\n"
        f"    With {plink_threads} threads.\n"
        f"    Adjust --time accordingly.\n"
        f"\n"
        f"  After all jobs finish, run:\n"
        f"    {next_step}"
    )

    return {
        "commands_file": commands_file,
        "script_file": script_file,
        "n_cells": n_cells,
        "submit_cmd": submit_cmd,
        "next_step": next_step,
    }


def prepare_scprs_data(
    adata: AnnData,
    labels: pd.Series,
    gwas_file: str | Path,
    prefix: str,
    prs_dir: str | Path,
    p_thresholds: list[float] | None = None,
    r2_thresholds: list[float] | None = None,
    n_pcs_knn: int = 20,
    knn_n_neighbors: int = 25,
    overwrite: bool = False,
) -> str:
    """
    Prepare scPRS input data from pre-computed PLINK score files.

    Reads PLINK score files generated by the SLURM array job
    (:func:`write_slurm_array_job`) into a ``(n_samples, n_cells, 21)``
    PRS matrix, builds the kNN graph, and serialises ``(X, y, edge)`` as a
    pickle for :func:`run_scprs`.

    Parameters
    ----------
    adata : AnnData
        Reference scATAC-seq AnnData. LSI embedding should be in
        ``adata.obsm["X_lsi"]`` (preferred) or ``adata.obsm["X_pca"]``.
    labels : pd.Series
        Binary case/control labels (1=case, 0=control) indexed by donor ID,
        matching the sample IDs in the PLINK ``.fam`` file.
    gwas_file : str or Path
        Post-QC GWAS summary statistics (tab-separated). Only used to
        determine p-value thresholds if not supplied explicitly.
    prefix : str
        Output prefix for all files.
    prs_dir : str or Path
        Directory containing pre-computed PLINK score files in the layout
        ``{prs_dir}/{r2}/{cell}.{p}.profile`` produced by the SLURM array
        job. Run :func:`write_slurm_array_job` first, then pass the same
        ``prs_dir`` here.
    p_thresholds : list of float, optional
        C+T p-value thresholds. Default: ``[1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5]``
    r2_thresholds : list of float, optional
        C+T LD r² thresholds. Default: ``[0.1, 0.3, 0.5]``
    n_pcs_knn : int, default=20
        Number of LSI/PCA dimensions used for kNN graph.
    knn_n_neighbors : int, default=25
        k for kNN graph (paper default: 25).
    overwrite : bool, default=False
        Re-run even if the output pickle already exists.

    Returns
    -------
    str
        Path to the generated pickle file ``{prefix}_scprs_data.pkl``.
    """
    pkl_path = f"{prefix}_scprs_data.pkl"
    if os.path.isfile(pkl_path) and not overwrite:
        logger.info(f"Found existing pickle at {pkl_path}. Use overwrite=True to recompute.")
        return pkl_path

    for col in ["chrom", "start", "end"]:
        if col not in adata.var.columns:
            raise ValueError(
                f"adata.var must contain '{col}'. " "Please annotate peaks with chromosome, start, end coordinates."
            )

    p_thresholds = p_thresholds or [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5]
    r2_thresholds = r2_thresholds or [0.1, 0.3, 0.5]

    edge = _build_knn_graph(adata, k=knn_n_neighbors, n_pcs=n_pcs_knn)

    prs_dir = str(prs_dir)
    logger.info(f"Using pre-computed PRS score files from {prs_dir}.")

    X, d_sample, d_cell, _, _ = _read_prs_score_files(
        prs_dir=prs_dir,
        r2_thresholds=r2_thresholds,
        p_thresholds=p_thresholds,
        labels=labels,
    )

    y = np.array(
        [labels.loc[sid] if sid in labels.index else 0 for sid in sorted(d_sample, key=d_sample.get)],
        dtype=np.float32,
    )

    with open(pkl_path, "wb") as f:
        pickle.dump((X, y, edge), f)
    logger.info(f"Saved scPRS data to {pkl_path}  X={X.shape}  y={y.shape}  edges={edge.shape}")
    return pkl_path


def run_scprs(
    data_path: str | Path,
    prefix: str,
    n_epoch: int = 350,
    n_repeats: int = 100,
    layer: int = 3,
    l1: float = 0.0,
    l2: float = 0.0,
    lgraph: float = 0.0,
    device: str = "cuda:0",
    save_model: bool = False,
) -> pd.DataFrame:
    """
    Train the scPRS GNN model (adaption of ``main.py`` (https://github.com/szhang1112/scPRS/blob/main/main.py)).

    Parameters
    ----------
    data_path : str or Path
        Pickle produced by :func:`prepare_scprs_data`.
        Must contain ``(X, y, edge)`` as in the original repo.
    prefix : str
        Prefix for output files.
    n_epoch : int, default=350
        Training epochs per repeat (paper default).
    n_repeats : int, default=100
        Number of random seeds (paper default).
    layer : int, default=3
        Number of GNN layers.
    l1 : float, default=0.0
        L1 regularisation on cell-importance weights.
    l2 : float, default=0.0
        L2 regularisation on cell-importance weights.
    lgraph : float, default=0.0
        Graph Laplacian regularisation.
    device : str, default="cuda:0"
        PyTorch device. Falls back to CPU if CUDA is unavailable.
    save_model : bool, default=False
        Save model checkpoints to ``{prefix}_{seed}.pt``.

    Returns
    -------
    pd.DataFrame
        Cell weight matrix of shape ``(n_repeats, n_cells)``, matching
        the CSV output of ``main.py``.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch and torch_geometric are required for run_scprs. " "Install with: pip install torch torch_geometric"
        )

    import random

    import networkx as nx
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    logger.info(f"Loading data from {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    X_raw = data[0]  # (n_samples, n_cells, n_features)
    y_raw = data[1]  # (n_samples,)
    edge = data[2]  # (n_edges, 2)

    means = X_raw.mean(0, keepdims=True)
    stds = X_raw.std(0, keepdims=True)
    stds[np.isnan(stds)] = 1
    X_norm = (X_raw - means) / (stds + 1e-7)

    n_cells = X_norm.shape[1]

    graph = nx.Graph()
    for i in range(n_cells):
        graph.add_edge(i, i)
    for e0, e1 in edge:
        graph.add_edge(int(e0), int(e1))
    L_np = nx.normalized_laplacian_matrix(graph).toarray()

    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        device = "cpu"
    dev = torch.device(device)

    L_tensor = torch.FloatTensor(L_np).to(dev)
    y_tensor = torch.FloatTensor(y_raw).to(dev)

    edge_index = torch.LongTensor(edge).to(dev).T
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32).to(dev)

    all_weights = []
    y_train_np = y_raw

    for random_state in tqdm(range(n_repeats), desc="scPRS repeats"):
        torch.manual_seed(random_state)
        random.seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        X_tensor = torch.FloatTensor(X_norm).to(dev)

        model = _scPRSModel(
            dim_in=X_norm.shape[-1],
            n_cell=n_cells,
            n_gcn=layer,
        ).to(dev)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(n_epoch):
            # Oversample positives to balance classes
            y_id = pd.DataFrame(y_train_np).reset_index()
            n_neg = int((y_train_np == 0).sum())
            y_id_pos = y_id[y_id[0] == 1].sample(n=n_neg, replace=True, random_state=epoch)
            select_idn = np.array(list(y_id_pos["index"]) + list(y_id[y_id[0] == 0]["index"]))

            dataset = TensorDataset(X_tensor[select_idn], y_tensor[select_idn])
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

            for xx, yy in dataloader:
                optim.zero_grad()
                out = model(xx, edge_index, edge_weight)[:, 0]
                loss = F.binary_cross_entropy_with_logits(out, yy)
                loss2 = abs(model.pred.weight).mean() * l1 + (model.pred.weight**2).mean() * l2
                loss3 = model.pred.weight @ L_tensor @ model.pred.weight.T * lgraph / len(model.pred.weight)
                (loss + loss2 + loss3).backward()
                optim.step()

        if save_model:
            torch.save(model, f"{prefix}_{random_state}.pt")

        all_weights.append(model.pred.weight.detach().cpu().numpy())

    weights_arr = np.array([w[0] for w in all_weights])

    result_path = f"{prefix}_scprs_weights.csv"
    weights_df = pd.DataFrame(weights_arr)
    weights_df.index.name = "repeat_id"
    weights_df.to_csv(result_path)
    logger.info(f"Saved cell weights to {result_path}")

    return weights_df


def get_disease_relevant_cells(
    weights_df: pd.DataFrame,
    adata: AnnData,
    top_percentile: float = 85.0,
    fdr_threshold: float = 0.1,
    cell_type_col: str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify disease-relevant cells from scPRS model weights.

    Adaptation of ``util/analysis.py`` (https://github.com/szhang1112/scPRS/blob/main/util/analysis.py):

    1. For each repeat, compute the ``top_percentile``-th percentile of all
       cell weights as a scalar reference value.
    2. For each cell, one-sided t-test comparing its weight distribution
       across repeats against the per-repeat percentile values.
    3. Benjamini-Hochberg FDR correction (``method='n'`` as in the paper).

    Parameters
    ----------
    weights_df : pd.DataFrame
        Cell weight matrix (n_repeats c n_cells) from :func:`run_scprs`.
    adata : AnnData
        Original scATAC-seq AnnData whose ``obs_names`` correspond to
        cell indices in ``weights_df.columns``.
    top_percentile : float, default=85.0
        Percentile used to define the reference distribution per repeat.
        The paper uses the 85th percentile (top 15%).
    fdr_threshold : float, default=0.1
        Adjusted p-value threshold (BY correction, as in analysis.py).
    cell_type_col : str, optional
        ``adata.obs`` column with cell-type labels for enrichment analysis.

    Returns
    -------
    pd.DataFrame or tuple
        If ``cell_type_col`` is None: DataFrame with per-cell results.
        If ``cell_type_col`` is provided: tuple of (cell_results, ct_results).
    """
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import fdrcorrection

    weights = weights_df.values

    ref_per_repeat = np.array([np.percentile(weights[i], top_percentile) for i in range(weights.shape[0])])

    pvals = []
    tstats = []
    for ci in range(weights.shape[1]):
        cell_w = weights[:, ci]
        t, p = ttest_ind(cell_w, ref_per_repeat, alternative="greater")
        tstats.append(t)
        pvals.append(p)

    rejected, pvals_adj = fdrcorrection(pvals, method="n", alpha=fdr_threshold)

    cell_results = pd.DataFrame(
        {
            "mean_weight": weights.mean(axis=0),
            "tstat": tstats,
            "pval": pvals,
            "pval_adj": pvals_adj,
            "is_disease_relevant": rejected,
        },
        index=adata.obs_names,
    )

    if cell_type_col is None:
        return cell_results

    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"'{cell_type_col}' not found in adata.obs.columns.")

    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    cell_results["cell_type"] = adata.obs[cell_type_col].values
    sel = cell_results["is_disease_relevant"].values

    ct_rows = []
    for ct in adata.obs[cell_type_col].unique():
        ct_mask = (adata.obs[cell_type_col] == ct).values
        a = (ct_mask & sel).sum()
        b = (ct_mask & ~sel).sum()
        c = (~ct_mask & sel).sum()
        d = (~ct_mask & ~sel).sum()
        or_, p_ = fisher_exact([[a, b], [c, d]], alternative="greater")
        ct_rows.append({"cell_type": ct, "n_selected": a, "OR": or_, "pval": p_})

    ct_df = pd.DataFrame(ct_rows)
    _, ct_adj, _, _ = multipletests(ct_df["pval"], method="fdr_bh")
    ct_df["pval_adj"] = ct_adj
    ct_df["significant"] = ct_adj < fdr_threshold
    ct_df = ct_df.set_index("cell_type")

    return cell_results, ct_df


def run_scprs_pipeline(
    adata: AnnData,
    labels: pd.Series,
    gwas_file: str | Path,
    prefix: str,
    prs_dir: str | Path,
    cell_type_col: str | None = None,
    n_epoch: int = 350,
    n_repeats: int = 100,
    layer: int = 3,
    l1: float = 0.0,
    l2: float = 0.0,
    lgraph: float = 0.0,
    device: str = "cuda:0",
    p_thresholds: list[float] | None = None,
    r2_thresholds: list[float] | None = None,
    knn_n_neighbors: int = 25,
    n_pcs_knn: int = 20,
    top_percentile: float = 85.0,
    fdr_threshold: float = 0.1,
    overwrite: bool = False,
    save_model: bool = False,
) -> tuple:
    """
    End-to-end scPRS pipeline.

    Chains :func:`prepare_scprs_data` → :func:`run_scprs` →
    :func:`get_disease_relevant_cells`.

    Returns
    -------
    tuple
        ``(weights_df, cell_results)`` if ``cell_type_col`` is None,
        ``(weights_df, cell_results, ct_results)`` otherwise.
    """
    pkl_path = prepare_scprs_data(
        adata=adata,
        labels=labels,
        gwas_file=gwas_file,
        prefix=prefix,
        prs_dir=prs_dir,
        p_thresholds=p_thresholds,
        r2_thresholds=r2_thresholds,
        knn_n_neighbors=knn_n_neighbors,
        n_pcs_knn=n_pcs_knn,
        overwrite=overwrite,
    )

    weights_df = run_scprs(
        data_path=pkl_path,
        prefix=prefix,
        n_epoch=n_epoch,
        n_repeats=n_repeats,
        layer=layer,
        l1=l1,
        l2=l2,
        lgraph=lgraph,
        device=device,
        save_model=save_model,
    )

    out = get_disease_relevant_cells(
        weights_df=weights_df,
        adata=adata,
        top_percentile=top_percentile,
        fdr_threshold=fdr_threshold,
        cell_type_col=cell_type_col,
    )

    if cell_type_col is not None:
        cell_results, ct_results = out
        return weights_df, cell_results, ct_results
    return weights_df, out
