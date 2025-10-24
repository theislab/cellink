# API

## Linking Structure

```{eval-rst}
.. module:: cellink._core
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    _core.donordata.DonorData
```

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

<!-- TODO: add pp._basic?  -->

```{eval-rst}
.. module:: cellink.pp
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    pp.variant_qc
```

## DonorData

```{eval-rst}
.. module:: cellink._core
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    _core.DonorData
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
```

## Utils

```{eval-rst}
.. module:: cellink.utils
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    utils.column_normalize
    utils.gaussianize
```

## Plotting

```{eval-rst}
.. module:: cellink.pl
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    pl.basic_plot
    pl.BasicClass
```
