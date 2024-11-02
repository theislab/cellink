import traceback
import sys
import logging
import warnings
import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence, Callable

from .paths import SCDATA, DUMP, ANNOTATION
from cellink.io import read_plink
from cellink import DonorData
from cellink.tl import EQTLData, EQTLPipeline

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def load_scdata(sc_data_path: str, annotation_path: str):
    ## reading single cell data
    scdata = ad.read_h5ad(sc_data_path)
    ## reading annotation data
    annotation_df = pd.read_csv(annotation_path).loc[
        :, ["ensembl_gene_id", "start_position", "end_position", "chromosome_name"]
    ]
    annotation_df = annotation_df.loc[annotation_df.ensembl_gene_id.isin(scdata.var_names)]
    ## merging the scdata.var df with the annotations
    merged_df = pd.merge(scdata.var, annotation_df, left_index=True, right_on="ensembl_gene_id")
    merged_df = merged_df.rename(
        columns={"ensembl_gene_id": "Geneid", "start_position": "start", "end_position": "end", "chromosome_name": "chrom"}
    )
    merged_df.index = merged_df["Geneid"]
    scdata.var = merged_df
    return scdata

def load_chrom_gdata(chrom: str, data_root: Path):
    plink_file = data_root / f"OneK1K_imputation_post_qc_r2_08/plink/chr{chrom}.dose.filtered.R2_0.8"
    gdata = read_plink(plink_file)
    return gdata

def get_pbdata_transforms(transforms: Sequence[str]):
    ...

def get_pv_transforms(pv_transforms: dict[str, Callable]):
    ...

@hydra.main(config_path="config", config_name="eqtl")
def main(config):
    ## loading sc data 
    sc_data = load_scdata(config.paths.scdata_path, config.paths.annotation_path)
    ## loading genetics data for current chromosome
    gdata = load_chrom_gdata(config.run.target_chrom, config.paths.gdata_path)
    ## initializing donor data
    data = DonorData(adata=scdata, gdata=gdata, donor_key_in_sc_adata=config.data.donor_key_in_sc_adata)
    ## initializing eqtl data
    eqtl_data = EQTLData(
        data,
        n_sc_comps=config.data.n_sc_comps,
        n_genetic_pcs=config.data.n_genetic_pcs
        donor_key_in_scdata=config.data.donor_key_in_scdata,
        sex_key_in_scdata=config.data.sex_key_in_scdata,
        age_key_in_scdata=config.data.age_key_in_scdata,
        pseudobulk_aggregation_type=config.data.pseudobulk_aggregation_type,
        n_top_genes=config.data.n_top_genes,
        min_individuals_threshols=config.data.min_individuals_threshold
    )
    ## retrieving transforms for bulked single cell data
    transforms = get_pbdata_transforms(config.eqtl.transforms)
    pv_transforms = get_pv_transforms(config.eqtl.pv_transforms)
    ## running eqtl pipeline and reporting results on all variants
    eqtl = EQTLPipeline(
        eqtl_data,
        cis_window=config.eqtl.cis_window,
        transforms=transforms,
        pv_transforms=pv_transforms,
        mode=config.eqtl.mode,
        dump_results=config.eqtl.dump_results, 
        file_prefix=config.eqtl.file_prefix, 
        dump_dir=config.eqtl.dump_dir
    )
    results_df_all = eqtl.run(config.run.target_cell_type, config.run.target_chrom, config.run.cis_window)
    return 0

if __name__=="__main__":
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)