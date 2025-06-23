import logging
import sys

import numpy as np
import anndata as ad
import pandas as pd
from pandas_plink import write_plink1_bin
import xarray as xr

from cellink._core.data_fields import VAnn, DAnn

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

def to_plink(gdata: ad.AnnData, output_prefix: str = None, donor_id: str = DAnn.donor, donor_family_id: str = DAnn.donor_family, chrom: str = VAnn.chrom, pos: str = VAnn.pos, a0: str = VAnn.a0, a1: str = VAnn.a1):
    """
    Export genotype data from an AnnData object to PLINK1 binary format (.bed, .bim, .fam).

    Parameters
    ----------
    gdata : anndata.AnnData
        AnnData object containing genotype data in `.X`, with variants in `.var` and sample metadata in `.obs`.
    output_prefix : str
        Prefix for the output PLINK files. If it does not end with '.bed', the extension is appended automatically.
    donor_id : str, default DAnn.donor
        Column in `gdata.obs` representing sample (individual) IDs.
    donor_family_id : str, default DAnn.donor_family
        Column in `gdata.obs` representing family IDs. If not present, `donor_id` is used instead.
    chrom : str, default VAnn.chrom
        Column in `gdata.var` representing chromosome information.
    pos : str, default VAnn.pos
        Column in `gdata.var` representing base-pair positions.
    a0 : str, default VAnn.a0
        Column in `gdata.var` representing the reference allele.
    a1 : str, default VAnn.a1
        Column in `gdata.var` representing the alternate allele.

    Returns
    -------
    None
        Writes `.bed`, `.bim`, and `.fam` files to disk using the provided prefix.

    Notes
    -----
    - Uses `xarray` to construct a labeled data array of genotypes.
    - Internally calls `sgkit.io.plink.write_plink1_bin` for exporting to PLINK format.
    """
    
    if not output_prefix.endswith(".bed"):
        output_prefix += ".bed"

    xarr = xr.DataArray(
        gdata.X.astype("float32"),
        dims=("sample", "variant"),
        coords={
            "sample": gdata.obs.index.to_numpy(),
            "sample_id": ("sample", gdata.obs[donor_id]), 
            "IID": ("sample", gdata.obs[donor_id]),   
            "iid": ("sample", gdata.obs[donor_id]),  
            "family_id": ("sample", gdata.obs[donor_family_id] if donor_family_id in gdata.obs.columns else gdata.obs[donor_id]), 
            "fid": ("sample", gdata.obs[donor_family_id] if donor_family_id in gdata.obs.columns else gdata.obs[donor_id]), 
            "i": ("sample", range(len(gdata.obs))),    
            "variant": gdata.var.index.to_numpy(),
            "snp": ("variant", gdata.var.index),       
            "chrom": ("variant", gdata.var[chrom]),
            "cm": ("variant", [0.0] *len(gdata.var)),
            "chr": ("variant", gdata.var[chrom]),      
            "pos": ("variant", gdata.var[pos]),       
            "a0": ("variant", gdata.var[a0]),         
            "a1": ("variant", gdata.var[a1]),
            "i": ("variant", range(len(gdata.var))),         
        },
        name="genotypes"
    )
    write_plink1_bin(xarr, output_prefix)

def write_variants_to_vcf(gdata, out_file="variants.vcf"):
    """Write unique variants from gdata to vcf file for annotation

    Parameters
    ----------
    gdata : gdata
        gdata object
    out_file : str, optional
        output file. By default "variants.vcf"
    """
    logger.info(f"number of variants to annotate: {len(gdata.var)}")
    var_df = gdata.var.reset_index()[[VAnn.chrom, VAnn.pos, VAnn.index, VAnn.a0, VAnn.a1]]
    var_df[["QUAL", "FILTER", "INFO"]] = ". . .".split()
    logger.info(f"Writing variants to {out_file}")
    with open(out_file, "w") as f:
        # Write the VCF header
        f.write("##fileformat=VCFv4.0\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        var_df.to_csv(f, sep="\t", index=False, header=False)
