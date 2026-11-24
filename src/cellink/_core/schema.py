from __future__ import annotations

import hidata
import pandas as pd
import pandera.pandas as pa
from hidata import HIDATA_ENCODING_TYPE, HIDATA_ENCODING_VERSION, HiDataSchemaError
from pandera.pandas import Column, DataFrameSchema

from .data_fields import VAnn

__all__ = [
    "HIDATA_ENCODING_TYPE",
    "HIDATA_ENCODING_VERSION",
    "GENO_VAR_SCHEMA",
    "HiDataSchemaError",
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


def validate(hd, check_var: bool = True) -> bool:
    """Check a `HiData` object against cellink's schema.

    Runs :func:`hidata.validate`, which checks that the levels are consistent, and then
    the variant annotations of the group level.

    Parameters
    ----------
    hd : HiData
        The object to check.
    check_var : bool, default=True
        Validate `hd.G.var` against `GENO_VAR_SCHEMA` (chrom/pos/a0/a1 present,
        correctly typed). When `hd.G` is a MuData, the `var` of every modality with a
        `chrom` column is validated.

    Returns
    -------
    bool
        `True` if every requested check passes.

    Raises
    ------
    HiDataSchemaError
        If any check fails, with a message naming the field or level that failed.
    """
    hidata.validate(hd)
    if check_var:
        if hd.G_type == "table":
            raise HiDataSchemaError("hd.G is a group table without variants; pass check_var=False to skip it.")
        if hasattr(hd.G, "mod"):
            tables = {f"hd.G['{name}'].var": mod.var for name, mod in hd.G.mod.items() if VAnn.chrom in mod.var}
            if not tables:
                raise HiDataSchemaError(f"No modality of hd.G has a '{VAnn.chrom}' column in its var.")
        else:
            tables = {"hd.G.var": hd.G.var}
        for name, var in tables.items():
            try:
                GENO_VAR_SCHEMA.validate(var)
            except pa.errors.SchemaError as e:
                raise HiDataSchemaError(f"{name} failed schema validation: {e}") from e

    return True


def require_genotypes(hd):
    """Return the genotypes `hd.G` of a `HiData`.

    Raises
    ------
    ValueError
        If `hd.G` is a group table, which holds no genotypes.
    """
    if isinstance(hd.G, pd.DataFrame):
        raise ValueError("This function needs genotypes in hd.G, but hd.G is a group table without genotypes.")
    return hd.G
