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
# # Resolving association-test inputs from a `DonorData`
#
# `cellink.at.GWAS`, `cellink.at.Skat` and `cellink.at.StructLMM` take a `DonorData` or
# an `AnnData` and nothing else. Every input is a formula string resolved against that
# object, and the variants come from the object itself, so a region is chosen by
# subsetting rather than by passing a matrix. Two function families cross the levels:
#
# - `crepeat(x)` broadcasts a **donor-level** value down to every cell of that donor.
# - `dmean(x)`, `dfirst(x)`, `dmax(x)`, `dmedian(x)` aggregate a **cell-level** variable
#   up to one value per donor.
#
# The genotypes, donors and cell types below are real OneK1K; the expression is
# simulated, so each test can be checked against a known ground truth.

# %% [markdown]
# ## Load
#
# `get_onek1k()` downloads ~17 GB and needs `PLINK` and `vcf2zarr` on `PATH`, so this
# notebook is committed with its outputs (`nb_execution_mode = "off"` in `docs/conf.py`).

# %%
import anndata as ad
import numpy as np
import pandas as pd
from liftover import get_lifter

from cellink import DonorData
from cellink.at import GWAS, Skat, StructLMM, get_model_matrix
from cellink.resources import get_onek1k

rng = np.random.default_rng(0)
dd = get_onek1k()

# %% [markdown]
# ## The variant
#
# `rs1574036` is the memory-B lead eSNP for *EAF2* in OneK1K (Yazar et al., *Science*
# **376**, eabf3041, 2022): a gene expressed across all cell types but genetically
# controlled in only two of the study's fourteen -- immature/naive B and memory B. That
# "expressed everywhere, regulated in one lineage" shape is exactly the pattern
# simulated below, which is why this variant and this window. Its real dosages are used;
# the expression is invented, so nothing here reproduces the published result.
#
# ### Mind the genome build
#
# The OneK1K VCF is **GRCh38** while the published eSNP tables are **GRCh37**, so a
# published position finds nothing until it is lifted. Ignore the `pos_hg19` column
# `get_onek1k()` adds -- it lifts coordinates that are already hg38 and lands ~281 kb
# off. `variant_id` holds positions rather than rsIDs, so there is no rsID lookup.

# %%
RS1574036_HG19 = 121_483_254  # rs1574036, as published
snp_pos = get_lifter("hg19", "hg38", one_based=True)["3"][RS1574036_HG19][0][1]

hits = dd.G.var[(dd.G.var["chrom"].astype(str) == "3") & (dd.G.var["pos"] == snp_pos)]
assert len(hits) == 1, f"expected one variant at 3:{snp_pos} (GRCh38), found {len(hits)}"

# the raw variant name starts with a digit, so copy the dosage to a formula-safe column
dd.G.obs["rs1574036"] = ad.utils.asarray(dd.G[:, hits.index[0]].X).ravel().astype(float)

# genotype PCs arrive with stringified integer column names; rename them so the `@`
# family operator can expand `@gPC[1:5]`
dd.G.obsm["gPCs"].columns = [f"gPC_{i}" for i in range(1, dd.G.obsm["gPCs"].shape[1] + 1)]

# %% [markdown]
# ## Cell types
#
# The expression half is Azimuth-annotated and carries none of the study's 14 labels, so
# they are rebuilt here. `CD4_SOX4` and `CD8_S100B` are marker-defined and cannot be
# recovered; the other twelve can. Check `dd.C.obs[label_col].value_counts()` first --
# spellings differ between annotations.

# %%
label_col = "predicted.celltype.l2" if "predicted.celltype.l2" in dd.C.obs else "cell_type"
ONEK1K_MAP = {
    "CD4_NC": ["CD4 Naive", "CD4 TCM"],
    "CD4_ET": ["CD4 TEM", "CD4 CTL"],
    "CD8_NC": ["CD8 Naive", "CD8 TCM"],
    "CD8_ET": ["CD8 TEM"],
    "NK": ["NK"],
    "NK_R": ["NK_CD56bright"],
    "B_IN": ["B naive", "B intermediate"],
    "B_Mem": ["B memory"],
    "Plasma": ["Plasmablast"],
    "Mono_C": ["CD14 Mono"],
    "Mono_NC": ["CD16 Mono"],
    "DC": ["cDC1", "cDC2", "pDC", "ASDC"],
}
to_onek1k = {a: k for k, labels in ONEK1K_MAP.items() for a in labels}

dd.C.obs["cell_state"] = pd.Categorical(
    dd.C.obs[label_col].astype(str).map(to_onek1k).fillna("other"),
    categories=[*ONEK1K_MAP, "other"],
)
dd.C.obs["cell_state"].value_counts()

# %% [markdown]
# ## Simulate a phenotype
#
# Three genetic components, one per test:
#
# 1. a **burden** over the *cis* window, the same slope in every cell type -- `GWAS`
# 2. a **random genetic effect**, an independent draw per variant so signs cancel in a
#    burden but still contribute variance across the set -- `Skat`
# 3. a **genotype x cell-type** effect, drawn only in the B states -- `StructLMM`
#
# Each is a fraction of the cell-level variance; noise takes the remainder.

# %%
VE_BURDEN, VE_RANDOM, VE_INTERACTION = 0.10, 0.07, 0.03
EFFECT_SIGN = -1.0
B_STATES = ("B_IN", "B_Mem")
WINDOW_BP, N_SET = 25_000, 100

window = dd.G.var[
    (dd.G.var["chrom"].astype(str) == "3") & (dd.G.var["pos"].between(snp_pos - WINDOW_BP, snp_pos + WINDOW_BP))
].copy()
if len(window) > N_SET:  # a diffuse effect over thousands of variants is undetectable
    window = window.loc[(window["pos"] - snp_pos).abs().nsmallest(N_SET).index].sort_values("pos")

X_raw = ad.utils.asarray(dd.G[:, window.index].X).astype(float)
alt_freq = 0.5 * X_raw.mean(axis=0)  # `X` counts a1, so fold it to get the MINOR allele
informative = np.minimum(alt_freq, 1.0 - alt_freq) > 0
X_raw, variant_names = X_raw[:, informative], window.index[informative]


def unit(x):
    """Centre and scale, so a sqrt(VE) coefficient is exactly that fraction."""
    return (x - x.mean()) / x.std()


burden = unit(X_raw.sum(axis=1))  # common variants, so weight 1 each
dd.G.obs["burden"] = burden
X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
R = unit(X_std @ rng.normal(0, 1 / np.sqrt(X_raw.shape[1]), X_raw.shape[1]))

# %% [markdown]
# `gamma` is drawn over every state present, `"other"` included with zero: a state
# missing from the index would make the `map()` below return `NaN`, and those cells
# would silently have no phenotype. Noise takes the remainder of the **realised** slope,
# since the burden contributes `(beta + gamma)^2` and the cross term does not cancel.

# %%
present = [s for s in dd.C.obs["cell_state"].cat.categories if (dd.C.obs["cell_state"] == s).any()]
cell_types = [s for s in present if s != "other"]

gamma = pd.Series([rng.normal(0, np.sqrt(VE_INTERACTION)) if s in B_STATES else 0.0 for s in present], index=present)
slope = EFFECT_SIGN * np.sqrt(VE_BURDEN) + gamma
ve_noise = 1.0 - slope**2 - VE_RANDOM
assert (ve_noise > 0).all(), "realised slope leaves no room for noise"

donor_ids = dd.C.obs[dd.donor_id].astype(str)  # a Categorical would poison the arithmetic
states = dd.C.obs["cell_state"].astype(str)
per_cell = pd.DataFrame({"burden": burden, "R": R}, index=dd.G.obs_names.astype(str)).reindex(donor_ids)

dd.C.obs["sim_expr"] = (
    states.map(slope).to_numpy() * per_cell["burden"].to_numpy()
    + np.sqrt(VE_RANDOM) * per_cell["R"].to_numpy()
    + np.sqrt(states.map(ve_noise).to_numpy()) * rng.normal(0, 1, dd.C.n_obs)
)
assert dd.C.obs["sim_expr"].notna().all(), "some cells have no simulated phenotype"

# %% [markdown]
# ## The resolver on its own
#
# `get_model_matrix(data, formula, target_level=...)` is what every model's `data=` path
# calls. `sex` and `age` are stored per cell even though they are donor attributes, so a
# donor-level model has to collapse them; `@gPC[1:5]` expands to the first five PCs.

# %%
get_model_matrix(dd, "dfirst(sex) + dfirst(age) + @gPC[1:5]", target_level="donor").head()

# %% [markdown]
# The other direction: the donor genotype broadcast to every cell and crossed with the
# cell state. This is the formula the numpy interface could not express without tiling
# the genotype by hand and keeping it in sync with every cell filter.

# %%
get_model_matrix(dd, "crepeat(rs1574036) * cell_state", target_level="cell").columns.tolist()

# %% [markdown]
# ## `GWAS`: one test per cell type
#
# `dmean()` builds the pseudobulk phenotype inside the formula, over whatever cells the
# object holds -- so subsetting to a cell type is what makes it a per-cell-type mean. A
# derived vector such as the burden is not in `.X`, so it travels as its own `AnnData`.

# %%
MIN_CELLS, MIN_DONORS = 10, 100
COVARIATES = "dfirst(sex) + dfirst(age) + @gPC[1:5]"
counts = dd.C.obs.groupby([dd.donor_id, "cell_state"], observed=True).size().unstack(fill_value=0)

results, skipped = {}, {}
for state in cell_types:
    donors = counts.index[counts[state] >= MIN_CELLS]
    if len(donors) < MIN_DONORS:  # Plasma and DC lose most donors to the floor
        skipped[state] = len(donors)
        continue
    dd_s = dd[:, :, dd.C.obs["cell_state"].eq(state).to_numpy(), :].copy()
    dd_s = dd_s[donors, :, :, :].copy()

    tested = ad.AnnData(X=dd_s.G.obs[["burden", "rs1574036"]].to_numpy(dtype=float), obs=dd_s.G.obs)
    gwas = GWAS(Y="dmean(sim_expr)", F=COVARIATES, data=dd_s, target_level="donor")
    gwas.test_association(tested)

    pv = np.ravel(gwas.getPv())
    results[state] = {"n_donors": dd_s.G.n_obs, "p_burden": pv[0], "p_lead_snp": pv[1]}

print(f"tested {len(results)} cell types; skipped {skipped}")
pd.DataFrame(results).T

# %% [markdown]
# The burden is significant in **every** cell type, because its slope is non-zero
# everywhere -- which says nothing about cell-type specificity. Only the interaction
# below does. `p_lead_snp` is the honest unknown: rs1574036 is one variant inside a
# 100-variant burden, so how much signal it carries depends on real LD in the window.

# %% [markdown]
# ## `StructLMM`: genotype x cell type
#
# `E` is a formula like any other slot, resolved with no intercept since a constant
# environment carries no interaction. The burden is donor-level while the phenotype is
# cell-level, so it is handed over as a `DonorData` and StructLMM broadcasts it. The
# result is **one p-value per variant column**, not one per cell: cells are the rows, and
# the whole 12-state `E` is tested as a single variance component.

# %%
MAX_CELLS_PER_DONOR = 5  # a cell-level model over 12 types x 982 donors is too much

keep = dd.C.obs.loc[dd.C.obs["cell_state"].isin(cell_types)]
sampled = (
    keep.groupby([dd.donor_id, "cell_state"], observed=True)
    .apply(lambda df: df.sample(min(len(df), MAX_CELLS_PER_DONOR), random_state=0))
    .index.get_level_values(-1)
)
dd_i = dd[:, :, dd.C.obs_names.isin(sampled), :].copy()
# empty Categorical levels would give formulaic all-zero columns, i.e. a singular E
dd_i.C.obs["cell_state"] = dd_i.C.obs["cell_state"].cat.remove_unused_categories()


def burden_of(d):
    """Wrap the burden as the variant set of `d`; it is derived, so it is not in `.X`."""
    return DonorData(G=ad.AnnData(X=d.G.obs[["burden"]].to_numpy(dtype=float), obs=d.G.obs), C=d.C)


slmm = StructLMM(
    y="sim_expr",
    E="cell_state - 1",
    F="sex + age",  # already per cell, so no `crepeat` needed
    data=dd_i,
    target_level="cell",
)
p_all = slmm.interaction_test(burden_of(dd_i), exact=True)
p_all

# %% [markdown]
# That number says the slope varies *somewhere*, not where. Drop the two states `gamma`
# was drawn in and it should go flat: the tested variant's own persistent effect is
# already a fixed effect in the null, so only the variation across `E` is left. Print the
# realised `gamma` too -- it is one draw per B state, and below `|gamma| ~ 0.05` there is
# nothing to find at this sample size.

# %%
non_b = ~dd_i.C.obs["cell_state"].isin(B_STATES).to_numpy()
dd_nb = dd_i[:, :, non_b, :].copy()
dd_nb.C.obs["cell_state"] = dd_nb.C.obs["cell_state"].cat.remove_unused_categories()

slmm_nb = StructLMM(y="sim_expr", E="cell_state - 1", F="sex + age", data=dd_nb, target_level="cell")
p_nb = slmm_nb.interaction_test(burden_of(dd_nb), exact=True)

print(f"realised gamma: {gamma[list(B_STATES)].round(3).to_dict()}")
print(f"all 12 states: {p_all.ravel()[0]:.2e}    without B_IN/B_Mem: {p_nb.ravel()[0]:.3g}")

# %% [markdown]
# Set `VE_INTERACTION = 0` and re-run: every marginal test above stays exactly as
# significant while this interaction disappears. That is the difference between "this
# gene has an eQTL" and "this gene has a *cell-type-specific* eQTL".

# %% [markdown]
# ## `Skat`: the variant set
#
# The set is whatever is in the object, so the window is chosen by subsetting. `a=1, b=1`
# gives flat weights, matching the unweighted burden; the default `a=1, b=25` is a
# rare-variant weighting and wrong for a window of common variants.

# %%
dd_b = dd[:, :, dd.C.obs["cell_state"].eq("B_Mem").to_numpy(), :].copy()
dd_b = dd_b[counts.index[counts["B_Mem"] >= MIN_CELLS], :, :, :].copy()
dd_cis = dd_b[:, variant_names, :, :].copy()

Skat(a=1, b=1, min_threshold=1).run_test(Y="dmean(sim_expr)", F="dfirst(sex) + dfirst(age)", data=dd_cis)

# %% [markdown]
# ## Plain `AnnData`
#
# `target_level` exists only because a `DonorData` has two levels. A plain `AnnData` has
# one, so it is omitted -- and the aggregation functions are rejected there, which is why
# the donor-level frame is collapsed with `dmean`/`dfirst` first and handed over ready-made.
# The phenotype here is its own: an effect from rs1574036 and nothing else, so the number
# below rests on the variant rather than on its LD with the rest of the burden.

# %%
VE_SNP = 0.05
g = unit(dd_b.G.obs["rs1574036"].to_numpy(dtype=float))

donor = dd_b.G.obs[["rs1574036"]].copy()
donor["expr"] = EFFECT_SIGN * np.sqrt(VE_SNP) * g + np.sqrt(1.0 - VE_SNP) * rng.normal(0, 1, len(g))
donor["age"] = get_model_matrix(dd_b, "dfirst(age) - 1", target_level="donor").to_numpy().ravel()

donor_adata = ad.AnnData(X=donor[["rs1574036"]].to_numpy(dtype=float), obs=donor[["expr", "age"]])
gwas_adata = GWAS(Y="expr", F="age", data=donor_adata)  # no `target_level`: there is only one
gwas_adata.test_association(donor_adata)  # the variants are this object's `.X`
gwas_adata.getPv()

# %% [markdown]
# ## Things that bite on real data
#
# - **Names must be unique.** A bare name is searched across `dd.G` and `dd.C`, in
#   `obs`, `var`, `obsm`/`varm` and `X`; two matches raise `Key '...' is not unique`.
# - **`crepeat` is for donor-level variables only.** `sex`/`age` live in `dd.C.obs`, so
#   at cell level they are used directly -- `crepeat(sex)` raises there.
# - **Subsetting cells changes the donor set.** `dd[:, :, cells, :]` re-intersects `G`
#   against the donors still in `C`, so a donor with no cells left is removed, not left
#   as `NaN`. Check `dd.G.n_obs` after a cell filter.
# - **Genome build.** `dd.G.var["pos"]` is GRCh38; published tables are GRCh37.
# - **Donors power these tests, not cells.** The phenotype is one number per donor no
#   matter how many cells went into it.
