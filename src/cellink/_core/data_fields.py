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


@dataclass(frozen=True)
class CAnn:
    """Cell annotation fields in adata.obs and gdata.obs"""

    celltype: str = "celltype"


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
