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
# `cellink.at.GWAS`, `cellink.at.Skat` and `cellink.at.StructLMM` take a `DonorData` or an
# `AnnData` and nothing else. Phenotype, covariates and environments are formula strings
# evaluated against that object, and the variants are whatever the object holds, so you pick
# a region by subsetting instead of passing a matrix around.
#
# Two families of functions move values between the donor and cell levels:
#
# - `crepeat(x)` copies a donor-level value to each of that donor's cells.
# - `dmean(x)`, `dfirst(x)`, `dmax(x)` and `dmedian(x)` collapse a cell-level variable to
#   one value per donor.
#
# We use real OneK1K genotypes, donors and cell types, and simulate the expression on top.
# Because we know the true effects, we can check each test's answer.

# %% [markdown]
# ## Load
#
# `get_onek1k()` downloads about 17 GB and needs `PLINK` and `vcf2zarr` on your `PATH`. The
# docs build doesn't execute notebooks (`nb_execution_mode = "off"` in `docs/conf.py`), so
# the outputs you see were saved from a local run.

# %%
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from liftover import get_lifter

from cellink import DonorData
from cellink.at import GWAS, Skat, StructLMM, get_model_matrix
from cellink.resources import get_onek1k

rng = np.random.default_rng(0)
dd = get_onek1k()

# colours for the figures: B states in blue, everything else in grey
B_COLOR, OTHER_COLOR, TRUTH_COLOR = "#2a78d6", "#a3a29c", "#eb6834"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})

# %% [markdown]
# ## The variant
#
# In OneK1K (Yazar et al., *Science* 376, eabf3041, 2022), `rs1574036` is the lead memory-B
# eSNP for *EAF2*. The gene is expressed in every cell type, but the study found a genetic
# effect on it in only two of its fourteen: immature/naive B and memory B. That's the
# situation we simulate below, so it's a natural place to anchor the example. We use the
# real dosages at this variant. The expression is made up, so don't expect to reproduce the
# published numbers.
#
# ### Genome build
#
# The OneK1K VCF is on GRCh38 and the published eSNP tables are on GRCh37, so you have to
# lift a published position before looking it up. Don't use the `pos_hg19` column that
# `get_onek1k()` adds: it lifts coordinates that are already hg38 and ends up about 281 kb
# away. `variant_id` holds positions, not rsIDs, so you can't look a variant up by rsID
# either.

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
# The expression data comes with Azimuth labels, not the study's 14 cell types, so we map
# one onto the other. Twelve of the fourteen can be rebuilt this way. `CD4_SOX4` and
# `CD8_S100B` were defined by marker genes and have no Azimuth equivalent. Label spellings
# vary between annotation versions, so look at `dd.C.obs[label_col].value_counts()` before
# trusting the mapping.

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
# The simulated expression has three genetic parts, one aimed at each test:
#
# 1. A burden over the *cis* window with the same slope in every cell type. `GWAS` should
#    pick this up.
# 2. A random effect with an independent draw per variant. The signs cancel in a burden, but
#    the set as a whole still explains variance. That's what `Skat` looks for.
# 3. A genotype-by-cell-type effect, drawn only in the two B states. This is the target for
#    `StructLMM`.
#
# Each part explains a fixed share of the cell-level variance, and noise makes up the rest.

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
# We draw `gamma` for every state that has cells, and set it to zero outside the B states,
# `"other"` included. If a state were missing from the index, the `map()` below would
# return `NaN` for its cells and they would quietly end up with no phenotype. The noise
# variance is computed from the slope we actually drew: the burden contributes
# `(beta + gamma)^2`, and the cross term doesn't cancel.

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
print(f"realised gamma: {gamma[list(B_STATES)].round(3).to_dict()}")

# %% [markdown]
# Before running any test, here is what we simulated. Each point is one donor's mean
# expression in that cell type, plotted against their burden. The grey line in the B panels
# is the slope shared by all cell types. The gap between it and the blue fit is `gamma`.

# %%
pseudobulk = dd.C.obs.groupby([dd.donor_id, "cell_state"], observed=True)["sim_expr"].mean().unstack()
donor_burden = dd.G.obs["burden"].reindex(pseudobulk.index.astype(str))

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True, sharey=True)
for ax, state in zip(axes, ["CD4_NC", *B_STATES], strict=True):
    x, y = donor_burden.to_numpy(), pseudobulk[state].to_numpy()
    ok = np.isfinite(y)
    color = B_COLOR if state in B_STATES else OTHER_COLOR
    ax.scatter(x[ok], y[ok], s=8, alpha=0.35, color=color, linewidths=0)
    grid = np.linspace(x.min(), x.max(), 2)
    ax.plot(grid, np.polyval(np.polyfit(x[ok], y[ok], 1), grid), color=color, lw=2)
    if state in B_STATES:
        ax.plot(grid, EFFECT_SIGN * np.sqrt(VE_BURDEN) * grid, color=OTHER_COLOR, lw=2, ls="--")
    ax.set_title(f"{state}  (true slope {slope[state]:+.2f})", fontsize=10)
    ax.set_xlabel("cis burden (standardised)")
axes[0].set_ylabel("donor mean sim_expr")
fig.tight_layout()

# %% [markdown]
# ## The resolver on its own
#
# Every model's `data=` path calls `get_model_matrix(data, formula, target_level=...)`. You
# can call it yourself to see what a formula turns into. `sex` and `age` are donor
# attributes but they're stored on each cell, so a donor-level model has to collapse them
# with `dfirst`. `@gPC[1:5]` expands to the first five genotype PCs.

# %%
get_model_matrix(dd, "dfirst(sex) + dfirst(age) + @gPC[1:5]", target_level="donor").head()

# %% [markdown]
# Going the other way, `crepeat` copies the donor genotype to every cell so it can be
# crossed with the cell state. With the numpy interface you'd have to tile the genotype
# yourself and keep it in sync with every cell filter.

# %%
get_model_matrix(dd, "crepeat(rs1574036) * cell_state", target_level="cell").columns.tolist()

# %% [markdown]
# ## `GWAS`: one test per cell type
#
# `dmean()` builds the pseudobulk phenotype inside the formula from whatever cells the
# object holds. Subset the object to one cell type and you get that cell type's mean. The
# burden is a derived vector and doesn't live in `.X`, so we wrap it in its own `AnnData`.
#
# We only keep donors with at least 10 cells of the state, and only test states that still
# have 100 donors after that. The figure shows where that cut falls.

# %%
MIN_CELLS, MIN_DONORS = 10, 100
COVARIATES = "dfirst(sex) + dfirst(age) + @gPC[1:5]"
counts = dd.C.obs.groupby([dd.donor_id, "cell_state"], observed=True).size().unstack(fill_value=0)

n_eligible = (counts[cell_types] >= MIN_CELLS).sum().sort_values()
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.barh(
    n_eligible.index,
    n_eligible.to_numpy(),
    color=[B_COLOR if s in B_STATES else OTHER_COLOR for s in n_eligible.index],
    height=0.7,
)
ax.axvline(MIN_DONORS, color="#52514e", lw=1, ls="--")
ax.text(MIN_DONORS, 0, f"  {MIN_DONORS} donors", color="#52514e", fontsize=9, va="center")  # Plasma row, below the cut
ax.set_xlabel(f"donors with at least {MIN_CELLS} cells")
fig.tight_layout()

# %%
results, skipped = {}, {}
for state in cell_types:
    donors = counts.index[counts[state] >= MIN_CELLS]
    if len(donors) < MIN_DONORS:
        skipped[state] = len(donors)
        continue
    dd_s = dd[:, :, dd.C.obs["cell_state"].eq(state).to_numpy(), :].copy()
    dd_s = dd_s[donors, :, :, :].copy()

    tested = ad.AnnData(X=dd_s.G.obs[["burden", "rs1574036"]].to_numpy(dtype=float), obs=dd_s.G.obs)
    gwas = GWAS(Y="dmean(sim_expr)", F=COVARIATES, data=dd_s, target_level="donor")
    gwas.test_association(tested)

    pv, beta, se = (np.ravel(a) for a in (gwas.getPv(), gwas.getBetaSNP(), gwas.getBetaSNPste()))
    results[state] = {
        "n_donors": dd_s.G.n_obs,
        "beta_burden": beta[0],
        "se_burden": se[0],
        "p_burden": pv[0],
        "p_lead_snp": pv[1],
    }

print(f"tested {len(results)} cell types; skipped {skipped}")
gwas_res = pd.DataFrame(results).T
gwas_res

# %% [markdown]
# The burden comes out significant in every cell type. That's expected, since its slope
# isn't zero anywhere, and it's also why these p-values can't tell you whether the effect is
# cell-type specific. The effect sizes are more informative. The T, NK and myeloid estimates
# all land in the same place, and the two B states sit closer to zero, which is what a
# positive `gamma` on a negative slope should do.
#
# Every estimate, B states included, comes out about 0.03 below its true slope. The offset
# is the same in every cell type, so it comes from something shared rather than from the
# interaction. The likely cause is the random component `R`: it's built from the same
# variants as the burden, so in any one draw the two are slightly correlated.
#
# `p_lead_snp` depends on the real data. rs1574036 is one of the 100 variants in the burden,
# so how strong it looks here depends on its LD with the others.

# %%
order = gwas_res.index[::-1]
fig, ax = plt.subplots(figsize=(6, 4))
y = np.arange(len(order))
colors = [B_COLOR if s in B_STATES else OTHER_COLOR for s in order]
ax.errorbar(
    gwas_res.loc[order, "beta_burden"],
    y,
    xerr=1.96 * gwas_res.loc[order, "se_burden"].abs(),
    fmt="none",
    ecolor=colors,
    elinewidth=2,
)
ax.scatter(gwas_res.loc[order, "beta_burden"], y, color=colors, s=40, zorder=3, label="estimated")
ax.scatter(slope[order], y, marker="|", s=200, color=TRUTH_COLOR, zorder=4, label="true slope")
ax.axvline(EFFECT_SIGN * np.sqrt(VE_BURDEN), color=OTHER_COLOR, lw=1, ls="--")
ax.set_yticks(y, order)
ax.set_xlabel("burden effect on donor mean (95% CI)")
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.tight_layout()

# %% [markdown]
# ## `StructLMM`: genotype by cell type
#
# `E` is a formula like the other inputs. We drop the intercept because a constant
# environment can't carry an interaction. The phenotype is per cell and the burden is per
# donor, so we pass the burden as a `DonorData` and StructLMM copies it to each cell.
#
# You get one p-value per variant column, not one per cell. The cells are the rows, and all
# 12 columns of `E` are tested together as a single variance component.

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
# A small p-value here means the slope differs between cell types, but it doesn't say which
# ones. To find out, drop the two B states and test again. The variant's shared effect is
# already a fixed effect in the null model, so with the B cells gone there should be nothing
# left to detect.
#
# Keep the realised `gamma` printed above in mind. It's a single random draw per B state, and
# if both draws land below about 0.05 in absolute value, this sample is too small to see the
# interaction at all.

# %%
non_b = ~dd_i.C.obs["cell_state"].isin(B_STATES).to_numpy()
dd_nb = dd_i[:, :, non_b, :].copy()
dd_nb.C.obs["cell_state"] = dd_nb.C.obs["cell_state"].cat.remove_unused_categories()

slmm_nb = StructLMM(y="sim_expr", E="cell_state - 1", F="sex + age", data=dd_nb, target_level="cell")
p_nb = slmm_nb.interaction_test(burden_of(dd_nb), exact=True)

print(f"realised gamma: {gamma[list(B_STATES)].round(3).to_dict()}")
print(f"all 12 states: {p_all.ravel()[0]:.2e}    without B_IN/B_Mem: {p_nb.ravel()[0]:.3g}")

# %% [markdown]
# Try setting `VE_INTERACTION = 0` and running everything again. The marginal tests above
# stay just as significant, and this interaction goes away. The first set of tests tells you
# the gene has an eQTL. Only this one tells you the eQTL is cell-type specific.

# %% [markdown]
# ## `Skat`: the variant set
#
# `Skat` tests whatever variants are in the object, so you choose the window by subsetting.
# `a=1, b=1` gives every variant the same weight, which matches our unweighted burden. The
# default `a=1, b=25` upweights rare variants and is the wrong choice for a window of common
# ones.

# %%
dd_b = dd[:, :, dd.C.obs["cell_state"].eq("B_Mem").to_numpy(), :].copy()
dd_b = dd_b[counts.index[counts["B_Mem"] >= MIN_CELLS], :, :, :].copy()
dd_cis = dd_b[:, variant_names, :, :].copy()

Skat(a=1, b=1, min_threshold=1).run_test(Y="dmean(sim_expr)", F="dfirst(sex) + dfirst(age)", data=dd_cis)

# %% [markdown]
# ## Plain `AnnData`
#
# `target_level` is only needed because a `DonorData` has two levels. A plain `AnnData` has
# one, so you leave it out. The aggregation functions aren't allowed there either, so we
# build the donor-level table with `dmean`/`dfirst` first and pass it in ready to use.
#
# This example gets its own phenotype, driven by rs1574036 alone. That way the result
# depends on the variant itself, not on its LD with the rest of the window.

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
# Here is the same thing as a picture: donor expression split by rs1574036 dosage.

# %%
dosage = donor["rs1574036"].round().astype(int)
groups = [donor.loc[dosage == k, "expr"].to_numpy() for k in sorted(dosage.unique())]

fig, ax = plt.subplots(figsize=(4.5, 3.4))
parts = ax.violinplot(groups, positions=range(len(groups)), showextrema=False, widths=0.8)
for body in parts["bodies"]:
    body.set_facecolor(B_COLOR)
    body.set_alpha(0.3)
ax.plot(range(len(groups)), [grp.mean() for grp in groups], "o-", color=B_COLOR, lw=2, ms=8)
ax.set_xticks(
    range(len(groups)), [f"{k}\n(n={len(grp)})" for k, grp in zip(sorted(dosage.unique()), groups, strict=True)]
)
ax.set_xlabel("rs1574036 dosage")
ax.set_ylabel("simulated donor expression")
fig.tight_layout()
