# API

## Input-Output

```{eval-rst}
.. module:: cellink.io
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    io.from_sgkit_dataset
    io.read_plink
    io.read_bgen
    io.read_sgkit_zarr
    io.write_variants_to_vcf
```

## Preprocessing

```{eval-rst}
.. module:: cellink.pp
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    pp.variant_qc
```

## DonorData

```{eval-rst}
.. module:: cellink
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    DonorData
```

## Tools

```{eval-rst}
.. module:: cellink.tl
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    tl.get_snp_df
    tl.run_favor
    tl.run_snpeff
    tl.run_vep
    tl.one_hot_encode_genotypes
    tl.dosage_per_strand
    tl.add_vep_annos_to_gdata
    tl.combine_annotations
    tl.aggregate_annotations_for_varm
    tl.run_burden_test
    tl.run_skat_test
    tl.beta_weighting
    tl.external.calculate_ld
    tl.external.run_jaxqtl
    tl.external.read_jaxqtl_results
    tl.external.run_mixmil
    tl.external.calculate_pcs
    tl.external.run_tensorqtl
    tl.external.read_tensorqtl_results
```

<!-- ## External Tools

```{eval-rst}
.. module:: cellink.tl.external
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated
    tl.external.calculate_ld
    tl.external.run_jaxqtl
    tl.external.read_jaxqtl_results
    tl.external.run_mixmil
    tl.external.calculate_pcs
    tl.external.run_tensorqtl
    tl.external.read_tensorqtl_results
``` -->

## Utils

```{eval-rst}
.. module:: cellink.utils
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    utils.column_normalize
    utils.gaussianize
```

## Resources

```{eval-rst}
.. module:: cellink.resources
.. currentmodule:: cellink

.. autosummary::
   :toctree: generated

   resources.get_1000genomes
   resources.get_onek1k
   resources.get_eqtl_catalog_dataset_associations
   resources.get_eqtl_catalog_datasets
   resources.get_gwas_catalog_studies
   resources.get_gwas_catalog_study
   resources.get_gwas_catalog_study_summary_stats
   resources.get_pgs_catalog_score
   resources.get_pgs_catalog_scores
   resources.get_1000genomes_ld_scores
   resources.get_1000genomes_ld_weights
```

## Plotting

```{eval-rst}
.. module:: cellink.pl
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    pl.locus
    pl.manhattan
    pl.qq
    pl.expression_by_genotype
    pl.volcano
```
