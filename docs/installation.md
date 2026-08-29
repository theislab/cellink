# Installation

cellink requires Python >=3.11 (tested on 3.11-3.13).

cellink is not yet published on PyPI (the `cellink` name there belongs to an unrelated
package; a future release will be published as `cellink-tools`). Until then, install
directly from GitHub:

## Development version

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/theislab/cellink.git@main
```

## Optional extras

The core install covers `DonorData` construction, I/O, and the tools built directly on its
dependencies. Most external-tool integrations under `cellink.tl.external` are opt-in extras.
Since there is no PyPI release yet, request them with the package's extras syntax appended
directly to the git URL (multiple extras can be combined, e.g. `[pgen,scooby]`):

```bash
pip install "cellink[pgen,scooby] @ git+https://github.com/theislab/cellink.git@main"
```

| Extra | Enables |
|---|---|
| `pgen` | PLINK2 PGEN/PVAR/PSAM genotype I/O (via `pgenlib`) |
| `datasets` | BGEN/VCF genotype I/O and the built-in dataset loaders (via `sgkit`/`bio2zarr`) |
| `at` | The formula-string resolver (`cellink.at.GWAS`), `StructLMM`, and `Skat` |
| `ml` | `MILDataset` and other `pytorch-lightning`-backed training utilities |
| `rvat` | Rare-variant association tooling |
| `scooby` | Scooby/Borzoi sequence-model variant-effect scoring |
| `scdrs` | Genotype-free cell-type disease relevance scoring (scDRS) |
| `ldsc` | Cell-type-specific LD score regression (via `gsmap`) |
| `tensorqtl` | cis-/trans-eQTL mapping via tensorQTL |
| `embpy` | Donor-level representation learning via `embpy` |
| `mixmil` | Mixed-effects multiple-instance learning |
| `scpoli` | scPoli-based embedding integration |
| `doc` | Building the documentation locally |
| `dev` | Contributor tooling (`pre-commit`, `twine`) |
| `test` | Running the test suite (`pytest`, `coverage`) |

A few external tools this package wraps (`jaxqtl`, LIVI, SAIGE-QTL, Ensembl VEP, SnpEff) are
not installable via pip; the corresponding `cellink.tl.external` function's own docstring and
error message link to that tool's own installation instructions.

## Next steps

Once installed, **[DonorData basics](tutorials/donordata_basics.ipynb)** is the place to
start: no analysis, just how to build a `DonorData` object from your own genotype/expression
data (or from the small synthetic dataset the tutorial generates on the fly), slice it, and
save it. From there, the **[tutorials](tutorials/index.md)** section covers each analysis
workflow, and the **[API reference](api/index.md)** documents every public function and class.
