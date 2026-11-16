from __future__ import annotations

import donordata
import pandas as pd
import pandera.pandas as pa
from donordata import DONORDATA_ENCODING_TYPE, DONORDATA_ENCODING_VERSION, DonorDataSchemaError
from pandera.pandas import Column, DataFrameSchema

from .data_fields import VAnn

__all__ = [
    "DONORDATA_ENCODING_TYPE",
    "DONORDATA_ENCODING_VERSION",
    "GENO_VAR_SCHEMA",
    "DonorDataSchemaError",
    "require_genotypes",
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


def validate(dd, check_var: bool = True) -> bool:
    """Check a `DonorData` object against cellink's schema.

    Runs :func:`donordata.validate`, which checks that the levels are consistent, and then
    the variant annotations of the donor level.

    Parameters
    ----------
    dd : DonorData
        The object to check.
    check_var : bool, default=True
        Validate `dd.D.var` against `GENO_VAR_SCHEMA` (chrom/pos/a0/a1 present,
        correctly typed). When `dd.D` is a MuData, the `var` of every modality with a
        `chrom` column is validated.

    Returns
    -------
    bool
        `True` if every requested check passes.

    Raises
    ------
    DonorDataSchemaError
        If any check fails, with a message naming the field or level that failed.
    """
    donordata.validate(dd)
    if check_var:
        if dd.D_type == "table":
            raise DonorDataSchemaError("dd.D is a donor table without variants; pass check_var=False to skip it.")
        if hasattr(dd.D, "mod"):
            tables = {f"dd.D['{name}'].var": mod.var for name, mod in dd.D.mod.items() if VAnn.chrom in mod.var}
            if not tables:
                raise DonorDataSchemaError(f"No modality of dd.D has a '{VAnn.chrom}' column in its var.")
        else:
            tables = {"dd.D.var": dd.D.var}
        for name, var in tables.items():
            try:
                GENO_VAR_SCHEMA.validate(var)
            except pa.errors.SchemaError as e:
                raise DonorDataSchemaError(f"{name} failed schema validation: {e}") from e

    return True


def require_genotypes(dd):
    """Return the genotypes `dd.D` of a `DonorData`.

    Raises
    ------
    ValueError
        If `dd.D` is a donor table, which holds no genotypes.
    """
    if isinstance(dd.D, pd.DataFrame):
        raise ValueError("This function needs genotypes in dd.D, but dd.D is a donor table without genotypes.")
    return dd.D
