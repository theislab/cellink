import logging
import os
import sys
import traceback
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import anndata as ad
import hydra
import pandas as pd
from omegaconf import DictConfig

from cellink import DonorData
from cellink.io import read_plink
from cellink.tl import EQTLData, EQTLPipeline, pbdata_transforms_dict, pv_transforms_dict

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
        columns={
            "ensembl_gene_id": "Geneid",
            "start_position": "start",
            "end_position": "end",
            "chromosome_name": "chrom",
        }
    )
    merged_df.index = merged_df["Geneid"]
    scdata.var = merged_df
    return scdata


def load_chrom_gdata(chrom: str, data_root: Path):
    plink_file = os.path.join(data_root, f"chr{chrom}.dose.filtered.R2_0.8")
    gdata = read_plink(plink_file)
    return gdata


def get_pbdata_transforms(transforms: Sequence[str], transforms_map: dict[str, Callable]) -> Sequence[Callable]:
    return [transforms_map[tr] for tr in transforms]


@hydra.main(config_path="./config", config_name="eqtl")
def main(config: DictConfig):
    ## loading sc data
    if config.run.verbose:
        logger.info("Loading scdata...")
    scdata = load_scdata(config.paths.scdata_path, config.paths.annotation_path)
    ## loading genetics data for current chromosome
    if config.run.verbose:
        logger.info("scdata loaded. Loading gdata...")
    gdata = load_chrom_gdata(config.run.target_chrom, config.paths.gdata_path)
    ## initializing donor data
    if config.run.verbose:
        logger.info("gdata loaded. Initializing DonorData...")
    data = DonorData(adata=scdata, gdata=gdata, donor_key_in_sc_adata=config.data.donor_key_in_scdata)
    ## initializing eqtl data
    if config.run.verbose:
        logger.info(f"Donor Data loaded. {data} Preparing data for EQTL Pipeline...")
    eqtl_data = EQTLData(
        data,
        n_sc_comps=config.data.n_sc_comps,
        n_genetic_pcs=config.data.n_genetic_pcs,
        donor_key_in_scdata=config.data.donor_key_in_scdata,
        sex_key_in_scdata=config.data.sex_key_in_scdata,
        age_key_in_scdata=config.data.age_key_in_scdata,
        pseudobulk_aggregation_type=config.data.pseudobulk_aggregation_type,
        n_top_genes=config.data.n_top_genes,
        min_individuals_threshold=config.data.min_individuals_threshold,
    )
    ## retrieving transforms for bulked single cell data
    if config.run.verbose:
        logger.info("EQTL Data ready, loading data transformations...")
    transforms = [pbdata_transforms_dict[transform] for transform in config.eqtl.transforms]
    pv_transforms = {transform: pv_transforms_dict[transform] for transform in config.eqtl.pv_transforms}
    ## running eqtl pipeline and reporting results on all variants
    if config.run.verbose:
        logger.info("Tranformations gathered, initializing the pipeline...")
    eqtl = EQTLPipeline(
        eqtl_data,
        cis_window=config.eqtl.cis_window,
        transforms=transforms,
        pv_transforms=pv_transforms_dict,
        mode=config.eqtl.mode,
        dump_results=config.eqtl.dump_results,
        file_prefix=config.eqtl.file_prefix,
        dump_dir=config.paths.dump_path,
    )
    if config.run.verbose:
        logger.info("Pipeline is ready, starting the run...")
    results_df_all = eqtl.run(config.run.target_cell_type, config.run.target_chrom)
    if config.run.verbose:
        logger.info("Run Finished!")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.info(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
