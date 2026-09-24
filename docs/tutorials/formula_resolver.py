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
# # Tutorial: resolving association-test inputs directly from `DonorData`
#
# `cellink.at.GWAS`, `cellink.at.Skat`, and `cellink.at.StructLMM` all take a
# `DonorData` or an `AnnData` and nothing else. Every input is named as a formula
# string or a bare column resolved against that object, and the variants come from
# the object itself -- `dd.G.X` or `adata.X` -- so a region is chosen by subsetting
# rather than by passing a matrix. A derived quantity such as a burden score is
# wrapped in an `AnnData` of its own before it can be tested.
#
# To show what that buys you, this tutorial sets up a **cell-type-specific
# eQTL**: a donor-level genotype that changes expression in one cell lineage
# and not in another. Writing that model down means crossing a per-donor value
# with a per-cell factor, which is precisely what the resolver's aggregation
# functions exist for:
#
# - `crepeat(x)` broadcasts a **donor-level** variable (a genotype) down to
#   every cell of that donor, so it can be crossed with a cell-level factor.
# - `dmean(x)`, `dfirst(x)`, `dmax(x)`, `dmedian(x)` aggregate a **cell-level**
#   variable up to one value per donor - pseudobulk expression, donor metadata.
#
# ## Real genotypes, simulated expression
#
# The genotypes, donors, cells and cell-type labels here are **real**: they come
# from OneK1K (Yazar et al., *Science* **376**, eabf3041, 2022) via
# `cellink.resources.get_onek1k()`. The **expression is simulated**, with an
# effect we plant ourselves.
#
# That is a deliberate choice, and it is not an attempt to reproduce the paper.
# Simulating the phenotype means we know the ground truth, so every section can
# *check* that the resolver built the model we intended rather than just assert
# it. Real genotypes give us honest allele frequencies, real donor structure and
# the real, badly unbalanced distribution of cells per donor - which is what
# actually governs power.
#
# The biology is *inspired by* one of the study's results. OneK1K reports
# *EAF2* (3q13.33) as an eGene in exactly **2 of its 14 cell types** - immature
# and naive B (B<sub>IN</sub>, lead eSNP rs7642303, Spearman rho = -0.568) and
# memory B (B<sub>Mem</sub>, lead rs1574036, rho = -0.493) - while the gene
# itself is expressed across all cell types. Both leads are in perfect LD
# (r2 = 1.000 in 1000G EUR), so that is one signal seen in two populations.
# We borrow that shape: a genotype effect in the B lineage, no effect in a
# T-cell comparator, and effect sizes of roughly that magnitude.
#
# We use the real rs1574036 dosages, and set the *expression* ourselves.

# %% [markdown]
# ## Loading OneK1K
#
# `get_onek1k()` downloads the published data (a ~4.4 GB CELLxGENE `.h5ad` plus
# a ~12.7 GB Zenodo VCF) and preprocesses the genotypes, which requires `PLINK`
# and `vcf2zarr` on `PATH`. That is far too heavy for a documentation build, so
# this notebook is **committed with its outputs pre-executed**
# (`nb_execution_mode = "off"` in `docs/conf.py`).

# %%
import anndata as ad
import numpy as np
import pandas as pd

from cellink import DonorData
from cellink.at import GWAS, Skat, StructLMM, get_model_matrix
from cellink.resources import get_onek1k

rng = np.random.default_rng(0)

dd = get_onek1k()
dd

# %% [markdown]
# ## Making the genotype and covariates formula-addressable
#
# Genotype PCs arrive as a `DataFrame` in `dd.G.obsm["gPCs"]` whose columns are
# stringified integers from the PLINK `.eigenvec` file. Renaming them to
# `gPC_1 ...` lets the `@` family operator expand them, so `@gPC[1:5]` becomes
# `gPC_1 + ... + gPC_5`.

# %%
n_pcs = dd.G.obsm["gPCs"].shape[1]
dd.G.obsm["gPCs"].columns = [f"gPC_{i}" for i in range(1, n_pcs + 1)]
dd.G.obsm["gPCs"].iloc[:3, :5]

# %% [markdown]
# ### Finding the variant: mind the genome build
#
# There are **two builds in play**, and they do not agree:
#
# - The OneK1K VCF (and therefore `dd.G.var["pos"]`) is **GRCh38**. Its
#   `variant_id` entries look like `chr3_121764407_A_G` - positions, not rsIDs,
#   so you cannot look a variant up by rsID at all.
# - The study's published eSNP tables are **GRCh37/hg19**, where rs1574036 is
#   at 3:121,483,254.
#
# Matching the published coordinate against `pos` therefore finds nothing. Lift
# it first, with the same `liftover` package `cellink` already depends on.
#
# Do **not** use the `pos_hg19` column that `get_onek1k()` adds: it applies an
# hg19 -> hg38 lifter to coordinates that are already hg38, so its values are
# neither build (they land ~281 kb away). Same for `id_hg19`.

# %%
from liftover import get_lifter

RS1574036_HG19 = 121_483_254  # as published in the OneK1K eSNP tables

lifter = get_lifter("hg19", "hg38", one_based=True)
RS1574036_POS = lifter["3"][RS1574036_HG19][0][1]  # -> 121_764_407 on GRCh38
print(f"rs1574036: hg19 3:{RS1574036_HG19} -> hg38 3:{RS1574036_POS}")

hits = dd.G.var[(dd.G.var["chrom"].astype(str) == "3") & (dd.G.var["pos"] == RS1574036_POS)]
assert len(hits) == 1, f"expected exactly one variant at 3:{RS1574036_POS} (GRCh38), found {len(hits)}"
snp_name = hits.index[0]  # '3_121764407_A_G'

# We copy the dosage into `dd.G.obs` under its rsID: the raw variant name starts
# with a digit and would otherwise need backticks in every formula.
dd.G.obs["rs1574036"] = ad.utils.asarray(dd.G[:, snp_name].X).ravel().astype(float)
hits[["chrom", "pos", "a0", "a1", "maf"]]

# %%
dd.G.obs["rs1574036"].value_counts().sort_index()

# %% [markdown]
# These are real calls, so the genotype classes are as unbalanced as they are
# in the cohort - the study reports an allele frequency of 0.636 for this
# variant, and the store agrees (MAF 0.361, imputation R2 0.94). Nothing below
# assumes balance.
#
# On orientation: `dd.G.X` counts copies of `a1`, which here is `G` - the same
# allele the study calls `A2` and reports its negative rho against. So effect
# signs are directly comparable, with no flip to undo.
#
# We keep this variant for the single-variant section, but note that the
# phenotype is **not** generated from it alone: it is one term in the gene-wide
# burden built below, and at MAF 0.36 the default rare-variant weights give it
# a small share. Expect the set test to beat the single-variant test here.

# %% [markdown]
# ## Reconstructing the OneK1K cell types
#
# The expression half of OneK1K comes from its CELLxGENE deposit, which was
# re-annotated with Azimuth and carries **no column with the study's 14
# cell-type labels**. They have to be rebuilt from `predicted.celltype.l2`.
#
# Twelve of the fourteen map cleanly. `CD4_SOX4` and `CD8_S100B` are
# marker-defined OneK1K clusters with no Azimuth counterpart, so they cannot be
# reconstructed from this file - the tutorial runs over the twelve it can
# recover, and says so rather than pretending otherwise.
#
# Print the labels you actually have before mapping: spellings differ between
# the Azimuth and Cell Ontology annotations, and this mapping is a judgement
# call you should check against the study's marker definitions.

# %%
label_col = "predicted.celltype.l2" if "predicted.celltype.l2" in dd.C.obs else "cell_type"
dd.C.obs[label_col].value_counts()

# %%
ONEK1K_MAP = {
    # OneK1K label: Azimuth predicted.celltype.l2 labels
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
    # not recoverable from Azimuth: CD4_SOX4, CD8_S100B
}

to_onek1k = {azimuth: onek1k for onek1k, labels in ONEK1K_MAP.items() for azimuth in labels}
labels = dd.C.obs[label_col].astype(str)
dd.C.obs["cell_state"] = pd.Categorical(
    labels.map(to_onek1k).fillna("other"),
    categories=[*ONEK1K_MAP, "other"],  # first level is the reference
)

unmapped = sorted(set(labels[dd.C.obs["cell_state"] == "other"]))
print(f"mapped {len(ONEK1K_MAP)} OneK1K types; {len(unmapped)} Azimuth labels left as 'other': {unmapped}")
dd.C.obs["cell_state"].value_counts()

# %% [markdown]
# ## Simulating the phenotype
#
# One fixed effect and **two random effects**, each parameterized by the
# fraction of cell-level variance it explains. For cell $i$ of donor $d$ in
# cell type $s$:
#
# $$y_i = \underbrace{\beta B_d}_{\text{fixed burden}}
#   + \underbrace{\sqrt{v^{\text{rand}}}\, R_d}_{\text{RE1: genotype}}
#   + \underbrace{\gamma_s B_d}_{\text{RE2: genotype} \times \text{cell type}}
#   + \sqrt{v^{\text{noise}}_s}\, \varepsilon_i$$
#
# **Fixed burden.** $B_d$ is an unweighted sum of the donor's dosages over
# every variant in the gene, standardized; $\beta = \sqrt{v^{\text{burden}}}$ is
# the same in **every** cell type. This is the effect a plain `GWAS` sees.
#
# **RE1 - genotype only.** $R_d = \sum_j x_{dj} b_j$ with
# $b_j \sim N(0, 1/m)$: each variant gets its own effect, so signs cancel in a
# burden but still contribute variance across the set. Identical in every cell
# type. **This is `Skat`'s target.**
#
# **RE2 - genotype x cell type.** $\gamma_s \sim N(0, v^{\text{int}})$, drawn
# per cell type and **non-zero only in `B_IN` and `B_Mem`**; it is exactly zero
# in the comparator, which is what makes the variance component identifiable.
# **This is `StructLMM`'s target.**
#
# The two random effects are deliberately different shapes: RE1 varies across
# *variants* and is shared by all cell types, RE2 varies across *cell types*
# and acts on the whole burden. A test for one is close to blind to the other.

# %%
VE_BURDEN = 0.10  # fixed burden slope, identical in every cell type
VE_RANDOM = 0.07  # RE1: genotype-only random effect   -> Skat
VE_INTERACTION = 0.03  # RE2: genotype x cell-type random effect -> StructLMM
EFFECT_SIGN = -1.0  # the study reports a negative rho for this locus

B_STATES = ("B_IN", "B_Mem")  # RE2 applies here and nowhere else

# %% [markdown]
# ### The variants, and the genotype scores

# %%
WINDOW_BP = 25_000  # tight: a diffuse effect over 1000s of variants is undetectable
N_SET = 100  # hard cap, so the set test's power is known rather than hoped for

window = dd.G.var[
    (dd.G.var["chrom"].astype(str) == "3")
    & (dd.G.var["pos"].between(RS1574036_POS - WINDOW_BP, RS1574036_POS + WINDOW_BP))
].copy()
if len(window) > N_SET:  # keep the N_SET nearest the lead, then restore position order
    window = window.loc[(window["pos"] - RS1574036_POS).abs().nsmallest(N_SET).index].sort_values("pos")

X_raw = ad.utils.asarray(dd.G[:, window.index].X).astype(float)  # dosages, 0/1/2
# `X` counts copies of a1, so the column mean halved is the ALT frequency. Fold it
# to get the *minor* allele frequency: an ALT-major variant would otherwise report
# a "MAF" above 0.5.
alt_freq = 0.5 * X_raw.mean(axis=0)
maf = np.minimum(alt_freq, 1.0 - alt_freq)
informative = maf > 0  # polymorphic in this cohort
variant_names = window.index[informative]
X_raw, maf = X_raw[:, informative], maf[informative]
m = X_raw.shape[1]
print(f"{m} polymorphic variants in +/-{WINDOW_BP // 1000} kb, MAF {maf.min():.3f}-{maf.max():.3f}")


def standardize(x: np.ndarray) -> np.ndarray:
    """Centre and scale to unit variance, so a sqrt(VE) coefficient is exact."""
    return (x - x.mean()) / x.std()


# the burden: every common variant weighted 1
burden = standardize(X_raw.sum(axis=1))
dd.G.obs["burden"] = burden  # donor-level, so formulas can `crepeat` it

# RE1: per-variant random effects, mixed signs
X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
R = standardize(X_std @ rng.normal(0, 1 / np.sqrt(m), m))

pd.DataFrame({"burden": burden, "RE1_genotype": R}, index=dd.G.obs_names).describe()

# %% [markdown]
# ### RE2: drawing the genotype x cell-type effect
#
# One $\gamma_s$ per cell type, shared by every cell of that type - which is
# what a one-hot environment matrix encodes - and zero outside the B states.
#
# Two things to keep in view. $\sqrt{v^{\text{int}}} = 0.173$ against
# $\beta = 0.316$, so RE2 is ~55% of the fixed slope: the B states can end up
# well above or below the comparator, and a draw can even come out near zero.
# And with only two non-zero cell types the *realized* variance is a poor
# estimate of the nominal one - detecting that it is non-zero is easy, pinning
# its value is not. Print the draw rather than trusting the parameter.

# %%
present = [s for s in dd.C.obs["cell_state"].cat.categories if (dd.C.obs["cell_state"] == s).any()]
cell_types = [s for s in present if s != "other"]  # the states we analyse

# gamma is drawn over EVERY state present, "other" included with gamma = 0. Leaving a
# state out of the index would make `states.map(...)` below return NaN for its cells,
# and their phenotype would be silently NaN.
gamma = pd.Series(
    [rng.normal(0, np.sqrt(VE_INTERACTION)) if s in B_STATES else 0.0 for s in present],
    index=present,
    name="gamma",
)
slope = EFFECT_SIGN * np.sqrt(VE_BURDEN) + gamma

# Noise takes the remainder of the REALISED slope, not of the nominal VE_BURDEN +
# VE_INTERACTION: the burden contributes (beta + gamma_s)^2, and the cross term
# 2*beta*gamma_s does not cancel. Subtracting the nominal fractions would leave the
# B states at ~0.89-1.13 total variance while every other state sat at 1.0.
ve_noise = (1.0 - slope**2 - VE_RANDOM).rename("ve_noise")
assert (ve_noise > 0).all(), "realised slope leaves no room for noise"

pd.DataFrame({"gamma": gamma, "slope": slope, "ve_noise": ve_noise})

# %% [markdown]
# ### Assembling the cell-level phenotype
#
# The donor-level terms have to be expanded to that donor's cells - the same
# operation `crepeat` performs inside a formula. And `dd.C.obs[donor_id]` is
# `Categorical`, so `.map()` would hand back a `Categorical` and silently
# poison the arithmetic; cast it first.

# %%
donor_ids = dd.C.obs[dd.donor_id].astype(str)
states = dd.C.obs["cell_state"].astype(str)
by_donor = pd.DataFrame({"burden": burden, "R": R}, index=dd.G.obs_names.astype(str))

per_cell = by_donor.reindex(donor_ids).to_numpy(dtype=float)
slope_cells = states.map(slope).to_numpy(dtype=float)  # beta + gamma_s
noise_cells = np.sqrt(states.map(ve_noise).to_numpy(dtype=float))

dd.C.obs["sim_expr"] = (
    slope_cells * per_cell[:, 0]  # fixed burden + RE2
    + np.sqrt(VE_RANDOM) * per_cell[:, 1]  # RE1
    + noise_cells * rng.normal(0, 1, dd.C.n_obs)  # cell noise
)
assert dd.C.obs["sim_expr"].notna().all(), "some cells have no simulated phenotype"
dd.C.obs.groupby("cell_state", observed=True)["sim_expr"].agg(["mean", "std", "size"])

# %% [markdown]
# ### What this implies at donor level
#
# Averaging a donor's cells shrinks **only** the noise term, by the number of
# cells; the burden and RE1 are constant per donor and do not shrink at all. So
# the donor-level correlation between burden and pseudobulk expression is
#
# $$\rho_s = \frac{|\beta + \gamma_s|}
#   {\sqrt{(\beta + \gamma_s)^2 + v^{\text{rand}} + v^{\text{noise}}_s / n_s}}$$
#
# There is no donor-level random effect in this model, so past a few dozen
# cells the noise term is nearly gone and these correlations run high - well
# above the ~0.5 the study reports for this locus. Adding a donor random term
# (or lowering `VE_BURDEN`) is what would pull them back down.

# %%
cells_per_donor = dd.C.obs.groupby([dd.donor_id, "cell_state"], observed=True).size().unstack(fill_value=0)

implied = {}
for state in cell_types:
    n_s = max(float(cells_per_donor[state].median()), 1.0)
    b_s = slope[state]
    implied[state] = {
        "median_cells": n_s,
        "gamma": gamma[state],
        "slope": b_s,
        "implied_rho": b_s / np.sqrt(b_s**2 + VE_RANDOM + ve_noise[state] / n_s),
    }
pd.DataFrame(implied).T

# %% [markdown]
# ## `get_model_matrix`: the resolver on its own
#
# `cellink.at.get_model_matrix(data, formula_str, target_level=...)` is what
# every model's `data=` path calls under the hood, and it is useful by itself
# for inspecting a design matrix before fitting anything.
#
# Note what `dfirst` is for: `sex` and `age` are donor attributes, but the
# CELLxGENE deposit stores them **per cell**, so a donor-level model has to
# collapse them. They are constant within a donor, so `dfirst` is the honest
# choice (`dmean` would give the same number).

# %%
X_donor = get_model_matrix(dd, "dfirst(sex) + dfirst(age) + @gPC[1:5]", target_level="donor")
X_donor.head()

# %% [markdown]
# And the other direction - the donor genotype broadcast down to every cell and
# crossed with the cell-level state. This formula is the point of the page:

# %%
X_cell = get_model_matrix(dd, "crepeat(rs1574036) * cell_state", target_level="cell")
X_cell.columns.tolist()

# %% [markdown]
# ## Testing per cell type
#
# The phenotype is generated per cell, but the eQTL is a donor-level question,
# so the model needs a per-donor mean *within one cell type*. `dmean()` does
# that inside the formula - no pre-aggregation step, no intermediate matrix.
#
# Restrict to one state at a time, and require a floor on cells per donor: a
# donor with three memory B cells contributes a mean that is mostly noise.

# %%
MIN_CELLS = 10
counts = dd.C.obs.groupby([dd.donor_id, "cell_state"], observed=True).size().unstack(fill_value=0)
COVARIATES = "dfirst(sex) + dfirst(age) + @gPC[1:5]"

MIN_DONORS = 100  # below this a per-type eQTL test is not worth reporting

results, skipped = {}, {}
for state in cell_types:
    cells = dd.C.obs["cell_state"].eq(state).to_numpy()
    donors = counts.index[counts[state] >= MIN_CELLS]
    if len(donors) < MIN_DONORS:
        # Rare types (Plasma, DC) lose most donors to the MIN_CELLS floor.
        # Record it rather than dropping it silently.
        skipped[state] = {"donors_with_min_cells": len(donors), "total_cells": int(counts[state].sum())}
        continue

    dd_s = dd[:, :, cells, :].copy()
    dd_s = dd_s[donors, :, :, :].copy()

    # The models take objects, not matrices, so a derived quantity is wrapped in an
    # AnnData of its own. Testing the burden (known ground truth) and the lead variant
    # (whose marginal effect depends on real LD in the window) in one call means two
    # columns, and `getPv()` returns one p-value per column.
    tested = ad.AnnData(X=dd_s.G.obs[["burden", "rs1574036"]].to_numpy(dtype=float), obs=dd_s.G.obs)
    gwas = GWAS(Y="dmean(sim_expr)", F=COVARIATES, data=dd_s, target_level="donor")
    gwas.test_association(tested)
    pv = np.ravel(gwas.getPv())

    y = get_model_matrix(dd_s, "dmean(sim_expr) - 1", target_level="donor").to_numpy().ravel()
    results[state] = {
        "n_donors": dd_s.G.n_obs,
        "n_cells": int(dd_s.C.n_obs),
        "implied_rho": implied[state]["implied_rho"],
        "rho_burden": float(np.corrcoef(np.asarray(tested.X)[:, 0], y)[0, 1]),
        "p_burden": float(pv[0]),
        "p_lead_snp": float(pv[1]),
    }
print(f"tested {len(results)} cell types; skipped {len(skipped)}: {skipped}")
pd.DataFrame(results).T

# %% [markdown]
# `rho_burden` should track `implied_rho` in every cell type, and `p_burden` is
# the verifiable one: the burden is exactly what the phenotype was built from.
#
# `p_lead_snp` is the honest unknown. rs1574036 is one variant inside a
# 100-variant burden, so its marginal effect depends on the real LD in the
# window - strong if it tags the block, weak if it does not. It is reported
# precisely because the tutorial cannot predict it.
#
# Every cell type shows an effect: the burden slope is non-zero everywhere.
# Only the interaction below says anything about cell-type specificity.

# %% [markdown]
# ## What the formula actually computed
#
# The aggregation functions are not magic: `dmean(sim_expr)` is a
# `groupby(donor_id).mean()` and `dfirst(sex)` a `groupby(donor_id).first()`. Worth
# checking once, because everything downstream trusts it.

# %%
cells = dd.C.obs["cell_state"].eq("B_Mem").to_numpy()
dd_b = dd[:, :, cells, :].copy()
dd_b = dd_b[counts.index[counts["B_Mem"] >= MIN_CELLS], :, :, :].copy()

by_donor = dd_b.C.obs.groupby(dd_b.donor_id, observed=True)
by_hand = by_donor["sim_expr"].mean().reindex(dd_b.G.obs_names).to_numpy(dtype=float)
by_formula = get_model_matrix(dd_b, "dmean(sim_expr) - 1", target_level="donor").to_numpy().ravel()
np.testing.assert_allclose(by_formula, by_hand)

meta = by_donor[["sex", "age"]].first().reindex(dd_b.G.obs_names)
np.testing.assert_allclose(
    get_model_matrix(dd_b, "dfirst(sex) - 1", target_level="donor").to_numpy().ravel(),
    meta["sex"].to_numpy(dtype=float),
)
print("dmean()/dfirst() match the equivalent groupby exactly.")

# %% [markdown]
# ## RE2: the genotype x cell-type variance component
#
# The per-cell-type tests above each see the same fixed burden, so a
# significant result in all of them says nothing about specificity. **RE2** is
# what does: $\gamma_s$ was drawn only for `B_IN` and `B_Mem`, so the genotype
# effect varies across cell types, and `StructLMM` estimates that variance.
#
# Every recovered cell type goes in. The ten non-B types are the $\gamma = 0$
# reference: without them there is nothing for the variance component to be
# measured against, and with twelve environments the variance is far better
# identified than it was from two.
#
# This is where `crepeat` becomes load-bearing: the genotype is one number per
# donor while the model is fitted over cells, so it has to be repeated across
# each donor's cells and kept in sync with every cell filter along the way.
# `y`, `E` and `F` are all formulas resolved against the data object. `E` is the
# environment design, so it takes no intercept: `"cell_state - 1"` gives one column
# per cell state.

# %%
# Twelve cell types x 982 donors is far too many cells to hand a cell-level
# model, so cap the cells per donor per type. This is a runtime concession and
# a safe one: the power lives in the donors, not the cells. Drop to a donor
# subset as well if it is still slow.
MAX_CELLS_PER_DONOR = 5

analysed = [s for s in cell_types if s in results or s in B_STATES]
state_cells = dd.C.obs.loc[dd.C.obs["cell_state"].isin(analysed)]
sampled = (
    state_cells.groupby([dd.donor_id, "cell_state"], observed=True)
    .apply(lambda df: df.sample(min(len(df), MAX_CELLS_PER_DONOR), random_state=0))
    .index.get_level_values(-1)
)
dd_i = dd[:, :, dd.C.obs_names.isin(sampled), :].copy()

# A Categorical keeps every level after subsetting, and empty levels would give
# formulaic all-zero columns - i.e. a singular E matrix.
dd_i.C.obs["cell_state"] = dd_i.C.obs["cell_state"].cat.remove_unused_categories()
print(dd_i.C.n_obs, "cells;", dd_i.C.obs["cell_state"].value_counts().to_dict())

# The interaction was planted on the burden, so that is the genotype term the test
# has to receive. Derived quantities travel as objects, and a cell-level phenotype
# needs a DonorData: StructLMM broadcasts each donor's value across their cells,
# which is what `crepeat` does inside a formula.
burden_dd = DonorData(
    G=ad.AnnData(X=dd_i.G.obs[["burden"]].to_numpy(dtype=float), obs=dd_i.G.obs),
    C=dd_i.C,
)

slmm = StructLMM(
    y="sim_expr",
    E="cell_state - 1",  # the environment is a formula like any other slot
    # sex/age are already per cell in the CELLxGENE deposit, so they need no
    # broadcast -- `crepeat` is only for variables that genuinely live in `dd.G`
    F="sex + age",
    data=dd_i,
    target_level="cell",
)
slmm.interaction_test(burden_dd, exact=True)

# %% [markdown]
# The burden slope is $\beta + \gamma_s$ with $\gamma$ drawn only in the B
# states, so a genotype x cell-type variance component is present by
# construction - `StructLMM` is estimating that variance, not a difference
# between two hand-picked cell types.
#
# Note it is a *different estimator* from the per-cell-type tests above: it
# works at the cell level with a random-effect null over `E`, whereas the
# equivalent donor-level statement is the spread of the per-cell-type slopes in
# the `implied` table. Expect the same conclusion, not the same p-value.
#
# The experiment worth running: set `VE_INTERACTION = 0` and re-run. Every
# gamma collapses to zero, so the burden slope becomes identical across cell
# types - and every marginal test stays exactly as significant as before while
# this interaction disappears. That is the distinction between "this gene has
# an eQTL" and "this gene has a *cell-type-specific* eQTL", and it is invisible
# to the per-cell-type tests.

# %% [markdown]
# ## Donors power the test, cells do not
#
# The phenotype is one number per donor, no matter how many cells went into it.
# Subsetting cells is comparatively harmless; subsetting donors is not - which
# is why a cohort of ~1000 donors exists in the first place.

# %%
cells = dd.C.obs["cell_state"].eq("B_Mem").to_numpy()
full = dd[:, :, cells, :].copy()
full = full[counts.index[counts["B_Mem"] >= MIN_CELLS], :, :, :].copy()

power = {}
for n in (100, 250, full.G.n_obs):
    subset = full[rng.choice(full.G.obs_names, size=n, replace=False), :, :, :].copy()
    gwas = GWAS(Y="dmean(sim_expr)", F=COVARIATES, data=subset, target_level="donor")
    gwas.test_association(ad.AnnData(X=subset.G.obs[["rs1574036"]].to_numpy(dtype=float), obs=subset.G.obs))
    power[n] = float(np.ravel(gwas.getPv())[0])
power

# %% [markdown]
# ## `Skat` over a *cis* window
#
# This is the test aimed at **component 2**, the random genetic effect: the
# per-variant effects have mixed signs, so they partly cancel in the burden but
# still contribute variance across the set.
#
# The window is kept tight because a variance component spread over thousands
# of variants gets diluted: at `VE_RANDOM = 0.07` and 982 donors, power is
# ~1.00 over 50-200 variants but falls to ~0.84 over 2000. `N_SET` caps the set
# so that power is a known quantity rather than a function of variant density.
# (At the earlier `VE_RANDOM = 0.02` the same cap was load-bearing: power over
# 2000 variants was only ~0.28.)
#
# `Skat.run_test` takes a variant *set* rather than one variant, and the set is
# whatever is in the object handed to it -- so a window is chosen by subsetting,
# the same way `GWAS` and `StructLMM` take theirs.

# %%
# The same window the phenotype was generated from, so the set effect is real.
# a=1, b=1 gives flat Beta weights, matching the unweighted burden we planted; the
# default a=1, b=25 is a rare-variant weighting and wrong for this window.
dd_cis = dd_b[:, variant_names, :, :].copy()
print(f"testing {dd_cis.G.n_vars} variants from the cis window")

skat = Skat(a=1, b=1, min_threshold=1)
skat.run_test(Y="dmean(sim_expr)", data=dd_cis)

# %% [markdown]
# Covariates work the same way, resolved against the same object:

# %%
skat.run_test(Y="dmean(sim_expr)", F="dfirst(sex) + dfirst(age)", data=dd_cis)

# %% [markdown]
# ## Plain `AnnData` inputs
#
# `target_level` exists only because a `DonorData` holds values at two levels. A
# plain `AnnData` has one, so it is omitted - and the aggregation functions are
# rejected there, since there is no second level to aggregate from.

# %%
donor_adata = ad.AnnData(
    X=dd_b.G.obs[["rs1574036"]].to_numpy(dtype=float),
    obs=pd.DataFrame(
        {
            "expr": by_hand,
            "sex": meta["sex"].to_numpy(dtype=float),
        },
        index=dd_b.G.obs_names,
    ),
)

gwas_adata = GWAS(Y="expr", F="sex", data=donor_adata)
gwas_adata.test_association(donor_adata)  # the variants are this object's .X
gwas_adata.getPv()

# %% [markdown]
# ## Things that bite on real data
#
# - **Names must be unique.** A bare name is searched across `dd.G` and
#   `dd.C`, in `obs`, `var`, `obsm`/`varm` and `X`. Two matches raise
#   `ValueError: Key '...' is not unique`, so avoid obs columns that collide
#   with gene or variant names, and suffix the columns of any donor-level
#   matrix you write into `dd.G.obsm`.
# - **`Categorical` donor IDs.** `.map()` on the `donor_id` column returns a
#   `Categorical`, which breaks arithmetic and makes `formulaic` one-hot encode
#   a numeric column. Cast with `.astype(str)` first.
# - **Genome build.** `dd.G.var["pos"]` is **GRCh38** for OneK1K, while the
#   study's published eSNP tables are GRCh37 - matching a published position
#   directly finds nothing. Lift it first. And ignore `pos_hg19`/`id_hg19`:
#   `get_onek1k()` builds them by applying an hg19 -> hg38 lifter to
#   already-hg38 coordinates, so they point ~281 kb off target.
# - **No rsIDs in the genotypes.** `variant_id` is `chr3_121764407_A_G`, i.e.
#   position and alleles, so a variant cannot be found by rsID.
# - **Cell-type labels are not the study's.** The CELLxGENE deposit is
#   Azimuth-annotated at finer resolution, with no mapping back to the 14
#   OneK1K clusters.
# - **Formula-unsafe names need backticks.** `dd.G.var_names` look like
#   `3_121483254_A_G` and start with a digit, so reference them as
#   `` `3_121483254_A_G` `` or copy the dosage into `obs` under its rsID.
# - **Subsetting cells changes the donor set.** `dd[:, :, cells, :]` rebuilds the
#   object and re-intersects `G` against the donors still present in `C`, so a donor
#   whose cells are all filtered out is *removed* from `dd.G` - there is no `NaN` row,
#   the row is gone. Check `dd.G.n_obs` after a cell filter. The loud consequence:
#   build a variant object from the pre-subset `dd.G`, test it against a model fitted
#   on the subset, and you get `variants cover N observations, but the model was
#   fitted on M`.
# - **`NaN` in an aggregated column silently shortens the design.** If every cell of a
#   donor carries `NaN` in the variable being aggregated, formulaic drops the row, so
#   `get_model_matrix` returns fewer rows than `dd.G.n_obs`. Compare the two.
