# API

## Linking Structure

```{eval-rst}
.. module:: cellink._core
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    _core.donordata.DonorData
    _core.data_fields.GAnn
    _core.data_fields.DAnn
    _core.data_fields.CAnn
    _core.data_fields.VAnn
```

## Simulation Data

```{eval-rst}
.. module:: cellink._core
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    _core.dummy_data.sim_adata
    _core.dummy_data.sim_adata_muon
    _core.dummy_data.sim_gdata
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
    io.to_plink
    io.read_donordata_objects
```

## Preprocessing

```{eval-rst}
.. module:: cellink.pp
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    pp.variant_qc
    pp.cell_level_obs_filter
    pp.donor_level_obs_filter
    pp.donor_level_var_filter
    pp.low_abundance_filter
    pp.missing_values_filter
    pp.log_transform
    pp.normalize
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
    tl.one_hot_encode_genotypes
    tl.dosage_per_strand
    tl.simulate_genotype_data_msprime
    tl.simulate_genotype_data_numpy
```

## Machine Learning

```{eval-rst}
.. module:: cellink.ml
.. currentmodule:: cellink

.. autosummary::
    :toctree: generated

    ml.DonorDataModel
    ml.DonorDataBaseModel
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
