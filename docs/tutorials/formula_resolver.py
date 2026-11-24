# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial: resolving association-test inputs directly from `HiData`
#
# `cellink.at.GWAS`, `cellink.at.Skat`, and `cellink.at.StructLMM` all accept
# either raw numpy arrays (their original interface, unchanged) or a `data=`
# object (a `HiData`, a plain `AnnData`, or a `pandas.DataFrame`) plus a
# formula string or bare column name per input. This tutorial demonstrates the
# `data=` path directly, and shows that it produces identical numeric results
# to the equivalent raw-numpy call.
#
# For a `HiData` specifically, two group/cell-level aggregation function
# families are available inside a formula string:
#
# - `group_mean(x)`, `group_max(x)`, `group_median(x)`, `group_first(x)`: aggregate a **cell-level**
#   variable `x` up to one value per group (for `target_level="group"`
#   formulas).
# - `crepeat(x)`: broadcast a **group-level** variable `x` down to every cell
#   of that group (for `target_level="cell"` formulas).

# %%
import numpy as np
import pandas as pd

from cellink import HiData
from cellink._core.dummy_data import sim_adata, sim_gdata
from cellink.at import GWAS, StructLMM, get_model_matrix

# %% [markdown]
# ## A simulated `HiData`
#
# `sim_gdata()` simulates a group-level genotype `AnnData` (`.G`); `sim_adata()`
# simulates a cell-level `AnnData` (`.C`), with the first few genes named
# `G0`, `G1`, ... For this tutorial we add a group-level phenotype and
# covariate directly onto `hd.G.obs`.

# %%
rng = np.random.default_rng(0)

hd = HiData(G=sim_gdata(), C=sim_adata(), group_id="donor_id")
hd.G.obs["age"] = rng.integers(20, 60, hd.G.n_obs).astype(float)
hd.G.obs["phenotype"] = rng.standard_normal(hd.G.n_obs)
hd

# %% [markdown]
# ## `get_model_matrix`: the resolver directly
#
# `cellink.at.get_model_matrix(data, formula_str, target_level=...)` is what
# every model's `data=` path calls under the hood. It's also directly useful
# on its own, e.g. to build a covariate matrix once and inspect it before
# passing it to a test.

# %%
# A group-level covariate matrix, aggregating a cell-level gene (G0) up to its
# per-group maximum with `group_max`, alongside a plain group-level column (age).
X_group = get_model_matrix(hd, "age + group_max(G0)", target_level="group")
X_group.head()

# %%
# The symmetric direction: broadcast a group-level variable (age) down to
# every cell of that group with `crepeat`.
X_cell = get_model_matrix(hd, "crepeat(age) + celltype", target_level="cell")
X_cell.head()

# %% [markdown]
# ## `GWAS` with `data=`
#
# `GWAS(Y=..., F=...)` still works exactly as before when `Y`/`F` are numpy
# arrays. Pass `data=` and `Y`/`F` as formula/column-name strings instead to
# resolve them from a `HiData`, `AnnData`, or `DataFrame` directly, with no
# manual `.to_numpy()` extraction needed.

# %%
gwas = GWAS(Y="phenotype", F="age", data=hd, target_level="group")
G = rng.standard_normal((hd.G.n_obs, 5))  # 5 variants, one column each
gwas.test_association(G)
gwas.getPv()

# %% [markdown]
# This gives the same result as resolving the arrays by hand and calling the
# original, numpy-only constructor:

# %%
Y_np = hd.G.obs[["phenotype"]].to_numpy()
F_np = np.column_stack([np.ones(hd.G.n_obs), hd.G.obs["age"].to_numpy()])
gwas_np = GWAS(Y=Y_np, F=F_np)
gwas_np.test_association(G)
np.testing.assert_allclose(gwas.getPv(), gwas_np.getPv())
print("data= and raw-numpy calls agree exactly.")

# %% [markdown]
# ## `GWAS`/`StructLMM` against a plain `AnnData` or `DataFrame`
#
# `target_level` is only needed for a `HiData` (which has values at both
# the group and cell level). A plain `AnnData` or `DataFrame` has one level,
# so it's omitted.

# %%
adata = sim_adata()
adata.obs["pheno"] = rng.standard_normal(adata.n_obs)
gwas_adata = GWAS(Y="pheno", data=adata)
gwas_adata.test_association(rng.standard_normal((adata.n_obs, 2)))
gwas_adata.getPv()

# %%
df = pd.DataFrame({"y": rng.standard_normal(200), "cov1": rng.standard_normal(200)})
s = StructLMM(y="y", E=rng.standard_normal((200, 3)), F="cov1", data=df)
s.y.shape, s.F.shape

# %% [markdown]
# ## `Skat`
#
# The same convention applies to `cellink.at.Skat.run_test`:

# %% tags=["skip-execution"]
from cellink.at import Skat

adata2 = sim_adata()
adata2.obs["pheno"] = rng.standard_normal(adata2.n_obs)
snp_cols = [f"snp{i}" for i in range(5)]
adata2.obs[snp_cols] = rng.integers(0, 3, size=(adata2.n_obs, 5)).astype(float)
skat = Skat(min_threshold=1)
skat.run_test(data=adata2, Y="pheno", X=" + ".join(snp_cols))
