import importlib
from typing import Any

from ._gsmap import format_gsmap_sumstats, load_gsmap_results
from ._jaxqtl import read_jaxqtl_results, run_jaxqtl
from ._joint_nmf import JointNMFWrapper
from ._ld import calculate_ld
from ._ldsc import (
    compute_ld_scores_with_annotations_from_bimfile,
    compute_ld_scores_with_annotations_from_donor_data,
    configure_ldsc_runner,
    estimate_celltype_specific_heritability,
    estimate_genetic_correlation,
    estimate_heritability,
    estimate_ld_scores_from_bimfile,
    estimate_ld_scores_from_donor_data,
    make_annot_from_bimfile,
    make_annot_from_donor_data,
    munge_sumstats,
)
from ._ldsc2magma import (
    genesets_dir_to_entrez_gmt,
    load_ensembl_to_entrez_map,
    run_magma_annotate,
    run_magma_gene_analysis,
    run_magma_gpa,
    run_magma_gsa,
    scores_to_covar,
    scores_to_gmt,
)
from ._livi import (
    LIVIRunner,
    configure_livi_runner,
    get_livi_runner,
    infer_livi,
    load_livi_results,
    run_livi_association_testing,
    save_livi_results,
    train_livi,
)
from ._magma import run_magma_pipeline
from ._pc import calculate_pcs
from ._saigeqtl import (
    configure_saigeqtl_runner,
    get_saigeqtl_runner,
    make_group_file,
    read_saigeqtl_results,
    run_saigeqtl,
)
from ._scdrs import run_scdrs
from ._sclinker import (
    compute_celltype_programs,
    compute_diseaseprogression_programs,
    compute_joint_nmf_programs,
    compute_nmf_programs,
)
from ._sclinker_utils import (
    bedgraph_to_snp_annotation,
    check_and_patch_ldsc_parse_bug,
    compute_escore,
    compute_ld_scores_for_sclinker,
    download_sclinker_enhancer_links,
    download_sclinker_references,
    genescores_to_100kb_bedgraph,
    genescores_to_abc_road_bedgraph,
    genescores_to_annotations,
    load_abc_links,
    load_gene_annotation,
    load_roadmap_links,
    load_sclinker_heritability_results,
    run_sclinker_heritability,
)
from ._scprs import (
    get_disease_relevant_cells,
    get_plink_commands_per_cell,
    prepare_scprs_data,
    run_scprs_pipeline,
    write_slurm_array_job,
)
from ._seismic import run_seismic
from ._seismic_torch import RegressionNLL, SparseScore, run_seismic_torch
from ._sldsc_utils import generate_gene_coord_file, generate_sldsc_genesets, get_magma_gene_loc, preprocess_for_sldsc
from ._tensorqtl import read_tensorqtl_results, run_tensorqtl

__all__ = [
    "read_jaxqtl_results",
    "run_jaxqtl",
    "calculate_ld",
    "calculate_pcs",
    "read_tensorqtl_results",
    "run_tensorqtl",
    "run_mixmil",
    "run_scdrs",
    "run_seismic",
    "run_seismic_torch",
    "SparseScore",
    "RegressionNLL",
    "load_gsmap_results",
    "format_gsmap_sumstats",
    "run_magma_pipeline",
    "configure_saigeqtl_runner",
    "get_saigeqtl_runner",
    "read_saigeqtl_results",
    "make_group_file",
    "run_saigeqtl",
    ###
    "compute_ld_scores_with_annotations_from_bimfile",
    "compute_ld_scores_with_annotations_from_donor_data",
    "configure_ldsc_runner",
    "estimate_celltype_specific_heritability",
    "estimate_genetic_correlation",
    "estimate_heritability",
    "estimate_ld_scores_from_bimfile",
    "estimate_ld_scores_from_donor_data",
    "make_annot_from_bimfile",
    "make_annot_from_donor_data",
    "munge_sumstats",
    ###
    "genesets_dir_to_entrez_gmt",
    "load_ensembl_to_entrez_map",
    "run_magma_annotate",
    "run_magma_gene_analysis",
    "run_magma_gpa",
    "run_magma_gsa",
    "scores_to_covar",
    "scores_to_gmt",
    ###
    "generate_gene_coord_file",
    "generate_sldsc_genesets",
    "get_magma_gene_loc",
    "preprocess_for_sldsc",
    ###
    "compute_celltype_programs",
    "compute_diseaseprogression_programs",
    "compute_nmf_programs",
    "compute_joint_nmf_programs",
    "bedgraph_to_snp_annotation",
    "compute_escore",
    "compute_ld_scores_for_sclinker",
    "download_sclinker_enhancer_links",
    "download_sclinker_references",
    "genescores_to_abc_road_bedgraph",
    "genescores_to_100kb_bedgraph",
    "genescores_to_annotations",
    "load_abc_links",
    "load_gene_annotation",
    "load_roadmap_links",
    "load_sclinker_heritability_results",
    "run_sclinker_heritability",
    "check_and_patch_ldsc_parse_bug",
    ###
    "JointNMFWrapper",
    ###
    "prepare_scprs_data",
    "get_plink_commands_per_cell",
    "write_slurm_array_job",
    "get_disease_relevant_cells",
    "run_scprs_pipeline",
    ###
    "LIVIRunner",
    "configure_livi_runner",
    "get_livi_runner",
    "train_livi",
    "infer_livi",
    "run_livi_association_testing",
    "save_livi_results",
    "load_livi_results",
    "train_livi_annbatch",
    "build_annbatch_collection",
    "read_g_from_dd_store",
    "CisGenotype",
    "LIVICisBatchAdapter",
    "AnnbatchLIVIDataModule",
]

_LIVI_ANNBATCH_NAMES = {
    "train_livi_annbatch",
    "build_annbatch_collection",
    "read_g_from_dd_store",
    "CisGenotype",
    "LIVICisBatchAdapter",
    "AnnbatchLIVIDataModule",
}


def __getattr__(name: str) -> Any:
    if name == "run_mixmil":
        try:
            module = importlib.import_module(f"{__name__}._mixmil")
            return getattr(module, name)
        except ImportError as e:
            raise ImportError(
                "Cannot import `run_mixmil`: this feature requires `torch` and `mixmil`. "
                "Install with:\n\n    pip install cellink[mixmil]"
            ) from e
    if name in _LIVI_ANNBATCH_NAMES:
        try:
            module = importlib.import_module(f"{__name__}._livi_annbatch")
            return getattr(module, name)
        except ImportError as e:
            raise ImportError(
                f"Cannot import `{name}`: this feature requires `torch`, `pytorch_lightning`, "
                "and `annbatch`. Install with:\n\n    pip install cellink torch pytorch_lightning annbatch"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
