from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

from .data_fields import VAnn

DONORDATA_ENCODING_TYPE = "donordata"
DONORDATA_ENCODING_VERSION = "0.1.0"

GENO_VAR_SCHEMA = DataFrameSchema(
    {
        VAnn.chrom: Column(pa.Object, nullable=False, coerce=True),
        VAnn.pos: Column(pa.Int64, nullable=False, coerce=True, checks=pa.Check.ge(0)),
        VAnn.a0: Column(pa.Object, nullable=False, coerce=True),
        VAnn.a1: Column(pa.Object, nullable=False, coerce=True),
    },
    strict=False,
    ordered=False,
)


class DonorDataSchemaError(ValueError):
    """Raised by `validate()` when a `DonorData` object does not conform to the
    documented on-disk schema (required `.var` fields, or donor-axis alignment)."""


def validate(dd, check_var: bool = True, check_donor_alignment: bool = True) -> bool:
    """Check a `DonorData` object against cellink's documented schema.

    Parameters
    ----------
    dd : DonorData
        The object to check.
    check_var : bool, default=True
        Validate `dd.G.var` against `GENO_VAR_SCHEMA` (chrom/pos/a0/a1 present,
        correctly typed).
    check_donor_alignment : bool, default=True
        Validate that `dd.G`'s and `dd.C`'s donor axes are the same set, in the
        same order. This is the invariant `DonorData._match_donors` establishes at
        construction time, made independently checkable here.

    Returns
    -------
    bool
        `True` if every requested check passes.

    Raises
    ------
    DonorDataSchemaError
        If any check fails, with a message naming the specific field or donor
        mismatch found.
    """
    if check_var:
        try:
            GENO_VAR_SCHEMA.validate(dd.G.var)
        except pa.errors.SchemaError as e:
            raise DonorDataSchemaError(f"dd.G.var failed schema validation: {e}") from e

    if check_donor_alignment:
        g_donors = _donor_ids(dd.G, dd.donor_id)
        c_donors = _donor_ids(dd.C, dd.donor_id)
        if len(g_donors) != len(c_donors):
            raise DonorDataSchemaError(
                f"dd.G and dd.C have different donor counts ({len(g_donors)} vs. {len(c_donors)}); "
                "DonorData's own construction should never allow this."
            )
        if list(g_donors) != list(c_donors):
            raise DonorDataSchemaError(
                "dd.G and dd.C donor axes are not in the same order; this is exactly the "
                "silent-mismatch failure mode DonorData's constructor exists to prevent, and "
                "should not be reachable through the public API."
            )

    return True


def _donor_ids(modality, donor_id: str):
    from mudata import MuData

    if isinstance(modality, MuData):
        return modality.obs_names
    if donor_id in modality.obs.columns:
        return modality.obs[donor_id].to_numpy()
    return modality.obs_names
