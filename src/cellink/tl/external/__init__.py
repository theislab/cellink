import importlib
from typing import Any

from ._jaxqtl import read_jaxqtl_results, run_jaxqtl
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
from ._ldsc2magma import genesets_dir_to_entrez_gmt
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
from ._seismic import run_seismic
from ._gsmap import load_gsmap_results, format_gsmap_sumstats
from ._magma import run_magma_pipeline
from ._sldsc_utils import generate_gene_coord_file, generate_sldsc_genesets, preprocess_for_sldsc
from ._tensorqtl import read_tensorqtl_results, run_tensorqtl
from ._sclinker import (
    compute_celltype_programs,
    compute_diseaseprogression_programs,
    compute_nmf_programs,
    compute_joint_nmf_programs,
)
from ._sclinker_utils import (
    run_sclinker_heritability,
    download_sclinker_enhancer_links,
    download_sclinker_references, 
    load_roadmap_links, 
    load_abc_links,
    load_gene_annotation,
    genescores_to_abc_road_bedgraph,
    genescores_to_100kb_bedgraph, 
    bedgraph_to_snp_annotation, 
    genescores_to_annotations,
    compute_ld_scores_for_sclinker,
    load_sclinker_heritability_results,
    compute_escore,
    check_and_patch_ldsc_parse_bug,
)
from ._joint_nmf import JointNMFWrapper

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
    "load_gsmap_results",
    "format_gsmap_sumstats",
    "run_magma_pipeline",
    "configure_saigeqtl_runner",
    "get_saigeqtl_runner",
    "read_saigeqtl_results",
    "make_group_file",
    "run_saigeqtl",
    ###
    "compute_celltype_programs",
    "compute_diseaseprogression_programs",
    "compute_nmf_programs",
    "compute_joint_nmf_programs",
    "geneprogram_to_bedgraph",
    "bedgraph_to_snp_annotation",
    "run_sclinker_sldsc",
    "load_sclinker_results",
    "compute_escore",
    "download_sclinker_references",
    "load_enhancer_links",
    "run_sclinker_pipeline",
    "download_sclinker_references", 
    "load_roadmap_links", 
    "load_abc_links",
    "load_gene_annotation",
    "genescores_to_abc_road_bedgraph",
    "genescores_to_100kb_bedgraph", 
    "bedgraph_to_snp_annotation", 
    "genescores_to_annotations",
    "compute_ld_scores_for_sclinker",
    "run_sclinker_sldsc",
    "load_sclinker_heritability_results",
    "compute_escore",
    "check_and_patch_ldsc_parse_bug",
    ###
    "JointNMFWrapper",
]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
