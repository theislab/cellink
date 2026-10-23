from __future__ import annotations

import pandera.pandas as pa
from donordata import DONORDATA_ENCODING_TYPE, DONORDATA_ENCODING_VERSION, DonorDataSchemaError
from donordata import validate as validate_donor_axis
from pandera.pandas import Column, DataFrameSchema

from .data_fields import VAnn

__all__ = [
    "DONORDATA_ENCODING_TYPE",
    "DONORDATA_ENCODING_VERSION",
    "DonorDataSchemaError",
    "GENO_VAR_SCHEMA",
    "validate",
]

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


def validate(dd, check_var: bool = True, check_donor_alignment: bool = True) -> bool:
    """Check a DonorData against the genetics schema of cellink.

    Parameters
    ----------
    dd
        The object to check.
    check_var
        Validate ``dd.D.var`` against ``GENO_VAR_SCHEMA``, which requires correctly typed ``chrom``,
        ``pos``, ``a0`` and ``a1`` columns. Skipped when ``D`` is a donor table without variants.
    check_donor_alignment
        Validate the donor axis with :func:`donordata.validate`: unique donors, every cell assigned
        to a donor of ``D`` and cells grouped in the donor order of ``D``.

    Returns
    -------
    bool
        ``True`` when every requested check passes.

    Raises
    ------
    DonorDataSchemaError
        Naming the field or donor mismatch that was found.
    """
    if check_var and dd.D_type != "table":
        try:
            GENO_VAR_SCHEMA.validate(dd.D.var)
        except pa.errors.SchemaError as e:
            raise DonorDataSchemaError(f"dd.D.var failed schema validation: {e}") from e
    if check_donor_alignment:
        validate_donor_axis(dd)
    return True
