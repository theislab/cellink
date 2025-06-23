from dataclasses import dataclass

DONOR_KEY = "donor_id"


@dataclass(frozen=True)
class GAnn:
    """Gene annotation fields in AnnData.var"""

    name: str = "name"  # gene name, name of adata.var.index
    start: str = "start"
    end: str = "end"
    chrom: str = "chrom"
    strand: str = "strand"


@dataclass(frozen=True)
class DAnn:
    """Donor annotation fields in adata.obs and gdata.obs"""

    donor: str = "donor_id"
    donor_family: str = "donor_family_id"
    DONOR_X_KEY: str = "donor_X"
    DONOR_LABELS_KEY: str = "donor_labels"
    DONOR_BATCH_KEY: str = "donor_batch"
    DONOR_CAT_COVS_KEY: str = "donor_extra_categorical_covs"
    DONOR_CONT_COVS_KEY: str = "donor_extra_continuous_covs"
    DONOR_INDICES_KEY: str = "donor_ind_x"
    DONOR_ID_KEY: str = "donor_id"


@dataclass(frozen=True)
class CAnn:
    """Cell annotation fields in adata.obs and gdata.obs"""

    celltype: str = "celltype"
    CELL_X_KEY: str = "cell_X"
    CELL_LABELS_KEY: str = "cell_labels"
    CELL_BATCH_KEY: str = "cell_batch"
    CELL_CAT_COVS_KEY: str = "cell_extra_categorical_covs"
    CELL_CONT_COVS_KEY: str = "cell_extra_continuous_covs"
    CELL_INDICES_KEY: str = "cell_ind_x"
    CELL_DONOR_KEY: str = "donor_id"


@dataclass(frozen=True)
class VAnn:
    """Variant annotation fields in GenoAnndata"""

    chrom: str = GAnn.chrom
    pos: str = "pos"
    a0: str = "a0"
    a1: str = "a1"
    asymb: str = "asymb"
    maf: str = "maf"
    contig: str = "contig"  # index for contig_id
    index: str = "snp_id"


@dataclass(frozen=True)
class AAnn:
    """Variant annotation fields in GenoAnndata Annotations"""

    chrom: str = VAnn.chrom
    pos: str = VAnn.pos
    a0: str = VAnn.a0
    a1: str = VAnn.a1
    index: str = VAnn.index
    gene_id: str = "gene_id"
    feature_id: str = "transcript_id"
    name_prefix: str = "variant_annotation"
    vep: str = "vep"
    snpeff: str = "snpeff"
