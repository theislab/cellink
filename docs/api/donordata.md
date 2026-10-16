## DonorData

`DonorData` comes from the [DonorData](https://github.com/theislab/donordata) package and is re-exported by
cellink. The donor level `D` can be genotypes as an AnnData, a MuData, or a plain table of
donor covariates, and an optional sample level `S` holds several samples per donor. The on-disk format is described
in the [DonorData documentation](https://github.com/theislab/donordata/blob/main/docs/io/spec.md).

```{eval-rst}
.. module:: cellink
.. currentmodule:: cellink

.. autosummary::
    :toctree: ../generated/

    DonorData
    concat
    validate
    register_donordata_namespace
    io.read_dd
    io.read_h5_dd
    io.read_zarr_dd
```

Everything of DonorData that cellink users need is available from cellink itself, so `import donordata` is not
required: `cellink.DonorData`, the readers in `cellink.io`, `cellink.pp` for the donor and cell filters, and
`cellink.ml.MILDataset` with `cellink.ml.mil_collate_fn`. These are the same objects as in DonorData.
