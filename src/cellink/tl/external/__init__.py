import importlib
from typing import Any

from ._ld import calculate_ld
from ._pc import calculate_pcs
from ._saigeqtl import (
    configure_saigeqtl_runner,
    get_saigeqtl_runner,
    make_group_file,
    read_saigeqtl_results,
    run_saigeqtl,
)
from ._tensorqtl import read_tensorqtl_results, run_tensorqtl
from ._jaxqtl import read_jaxqtl_results, run_jaxqtl
from ._scdrs import run_scdrs
from ._seismic import run_seismic
from ._magma import run_magma_pipeline
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
from ._sldsc_utils import generate_gene_coord_file, generate_sldsc_genesets, preprocess_for_sldsc
from ._ldsc2magma import load_ensembl_to_entrez_map, genesets_dir_to_entrez_gmt

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
    "run_magma_pipeline",
    "configure_saigeqtl_runner",
    "get_saigeqtl_runner",
    "read_saigeqtl_results",
    "make_group_file",
    "run_saigeqtl",
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
