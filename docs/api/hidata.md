## HiData

cellink works on {class}`hidata.HiData` objects from the [HiData](https://hidata.readthedocs.io) package,
available as `cellink.HiData`. In cellink the group level `hd.G` holds the genotypes, one row per donor. The
[HiData documentation](https://hidata.readthedocs.io) describes the object, its subgroup level and its
[file format](https://hidata.readthedocs.io/en/latest/io/spec.html); {doc}`io` reads and writes it with cellink's
genotype storage.

```{eval-rst}
.. module:: cellink
.. currentmodule:: cellink

.. autosummary::
    :toctree: ../generated/

    validate
```
