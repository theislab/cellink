# The `DonorData` on-disk format

`DonorData` objects can be written to HDF5 (`write_h5_dd`) or Zarr (`write_zarr_dd`) and
read back (`read_h5_dd`/`read_zarr_dd`/`read_dd`). This document describes that on-disk
representation as a versioned schema, not just a code convention, so a file can be checked
for conformance without importing cellink at all.

## Group layout

A `DonorData` file (HDF5 group or Zarr group) has this top-level structure:

```
/                               (root group)
  .attrs["encoding-type"]       "donordata"
  .attrs["encoding-version"]    "0.1.0"
  .attrs["donor_id"]            name of the donor-identifier column, e.g. "donor_id"
  .attrs["var_dims_to_sync"]    list of `.obsm` keys kept in sync between G and C
  G/                            genotype side (AnnData or MuData; see below)
  C/                            single-cell side (AnnData or MuData; see below)
  uns/<key>                     arbitrary unstructured data, one dataset per key
```

`G` and `C` are each written with their own AnnData/MuData `encoding-type`
(`"anndata"` or `"MuData"`), following those libraries' own on-disk conventions
unchanged; `DonorData` does not reinvent either format, only how the two are paired.

## Versioning policy

`encoding-version` follows semver-style `MAJOR.MINOR.PATCH`. A `MAJOR` bump means an
existing reader cannot correctly interpret the file without an explicit migration; `MINOR`
means new, optional structure was added (old readers still work, ignoring what they don't
recognize); `PATCH` means no structural change. The current version is `0.1.0`
(`cellink._core.schema.DONORDATA_ENCODING_VERSION`).

On read, `encoding-type` is checked strictly: a file whose top-level `encoding-type` is
present but not `"donordata"` is rejected with a clear error rather than silently misread.
`encoding-version` is checked permissively: a mismatch emits a warning (not an error) so
that files written by an older/newer cellink remain readable, since exact schema equality
is not required for the fields this reader actually looks at.

## Required `.var` fields (genotype side, `dd.G.var`)

Enforced by `cellink._core.schema.GENO_VAR_SCHEMA` (a `pandera` schema) and checked by
`cellink.validate(dd)`:

| field (`cellink._core.data_fields.VAnn`) | on-disk name | type | nullable |
|---|---|---|---|
| `chrom` | `chrom` | string | no |
| `pos` | `pos` | int64, >= 0 | no |
| `a0` | `a0` | string | no |
| `a1` | `a1` | string | no |

Additional columns (rsID, allele frequency, annotation, etc.) are permitted and ignored by
validation (`strict=False`); the four above are the only ones `DonorData`'s own downstream
tooling (`cellink.tl`, `cellink.at`) assumes are present.

## The `donor_id` contract

`DonorData(G, C, donor_id=...)` requires:

- `G.obs_names` to be the donor identifiers directly (one row per donor).
- `C.obs[donor_id]` to hold the same identifiers, potentially many rows (cells) per donor.

At construction, and again on every `G`/`C` reassignment through the public setters,
`DonorData._match_donors` intersects `G.obs_names` with `C.obs[donor_id].unique()`, restricts
both sides to that intersection, and reorders `C` so donors appear in exactly `G`'s own order.
This is the specific invariant `cellink.validate(dd, check_donor_alignment=True)` re-checks
independently: that `dd.G.obs_names` and the donor labels implied by `dd.C.obs[donor_id]`
(or `dd.C.obs_names`, for a MuData `C` already collapsed to one row per donor) are identical,
in the same order. A `DonorData` object that fails this check could only have been produced
by bypassing the public constructor/setters (e.g. mutating `._G`/`._C` directly), which is
unsupported.

## Checking conformance

```python
import cellink as cl

dd = cl.io.read_dd("donors.zarr")
cl.validate(dd)  # raises cellink.DonorDataSchemaError on any violation, else returns True
```

`validate()` does not require the file to have been read by cellink's own readers; it checks
the in-memory object's structure directly, so a `DonorData` assembled by hand from
independently-loaded `G`/`C` objects is checked exactly the same way as one round-tripped
through `write_h5_dd`/`read_h5_dd`.
