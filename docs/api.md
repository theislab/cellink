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
    io.read_sgkit_zarr
```

## Preprocessing

```{eval-rst}
.. module:: cellink.pp
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    pp.variant_qc
```

## Tools

```{eval-rst}
.. module:: cellink.tl
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    tl.get_snp_df
    tl.write_variants_to_vcf
    tl.run_vep
    tl.read_vep_annos
    tl.merge_annos_into_gdata
    tl.one_hot_encode_genotypes
    tl.dosage_per_strand
    tl.simulate_genotype_data_msprime
    tl.simulate_genotype_data_numpy
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
