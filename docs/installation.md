# Installation

cellink requires Python >=3.12 and is tested on Python 3.12, 3.13 and 3.14.

cellink is published on PyPI as `cellink-tools` (`cellink` on PyPI is an unrelated package):

```bash
pip install cellink-tools
```

## Development version

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/theislab/cellink.git@main
```

## Optional extras

The core install covers [HiData](https://hidata.readthedocs.io), genotype I/O, and the tools built directly on
their dependencies. Most external-tool integrations under `cellink.tl.external` are opt-in extras,
and several extras can be combined:

```bash
pip install "cellink-tools[pgen,datasets]"
```

| Extra | Enables |
|---|---|
| `pgen` | Reading PLINK 2 files into Zarr (`io.stream_pgen_to_zarr`) and writing them (`io.to_pgen`), via `pgenlib` |
| `datasets` | PLINK 1 (`io.read_plink`), VCF Zarr and BGEN genotype I/O and the built-in dataset loaders (via `sgkit`/`bio2zarr`) |
| `at` | The formula-string resolver (`cellink.at.GWAS`), `StructLMM`, and `Skat` |
| `rvat` | Rare-variant association tooling |
| `scooby` | Scooby/Borzoi sequence-model variant-effect scoring; scooby itself is installed from GitHub (see below) |
| `scdrs` | Genotype-free cell-type disease relevance scoring (scDRS) |
| `tensorqtl` | cis-/trans-eQTL mapping via tensorQTL (pinned to 1.0.10); q-values additionally need R with the Bioconductor `qvalue` package |
| `jaxqtl` | cis-/trans-eQTL mapping via jaxQTL (`cellink.tl.external.run_jaxqtl`) |
| `seismic_torch` | PyTorch implementation of seismic cell-type enrichment (`cellink.tl.external.run_seismic_torch`) |
| `sclinker` | sc-linker gene programs (`cellink.tl.external.compute_nmf_programs`, via `torchnmf`) |
| `scprs` | Single-cell polygenic risk scores with scPRS (`cellink.tl.external.run_scprs_pipeline`) |
| `embpy` | Donor-level representation learning via `embpy` |
| `mixmil` | Mixed-effects multiple-instance learning |
| `scpoli` | scPoli-based embedding integration |
| `doc` | Building the documentation locally |
| `dev` | Contributor tooling (`pre-commit`, `twine`) |
| `test` | Running the test suite (`pytest`, `coverage`) |

The dataloaders and the multiple instance learning model for the cells of each donor are in `hidata.ml` of HiData
(`pip install "hidata[ml]"`), see the [HiData tutorial](https://hidata.readthedocs.io/en/latest/tutorials/mil_model.html).

Some optional packages are not available for every Python version. On other versions, the
cellink functions that need them raise an error explaining why:

| Package | Supported Python |
|---|---|
| `scooby` | 3.12 |
| `tensorqtl` | 3.12, 3.13 |

scooby is not on PyPI. Install it from GitHub together with the `scooby` extra:

```bash
pip install "cellink-tools[scooby]" "scooby @ git+https://github.com/gagneurlab/scooby.git@061b7a2275a5f8f5ffb53ae6f7f2d34e3622a7ee"
```

`mixmil` needs `torch-scatter`, which has to be built against an installed PyTorch on every
Python version (`pip install torch-scatter --no-build-isolation` after installing `torch`).

A few external tools this package wraps (SAIGE-QTL, Ensembl VEP, SnpEff,
gsMap) are not installed with pip alongside cellink; gsMap is installed in its own environment
and called through its `gsmap` command line tool. The corresponding `cellink.tl.external`
function's docstring and error message link to that tool's installation instructions.

## Next steps

Once installed, **[HiData basics](tutorials/hidata_basics.ipynb)** is the place to
start: no analysis, just how to read genotype files into a `HiData` object, select from it
and save it, and HiData's own
[getting started](https://hidata.readthedocs.io/en/latest/tutorials/getting_started.html)
tutorial covers the object itself. From there, the **[tutorials](tutorials/index.md)** section covers each analysis
workflow, and the **[API reference](api/index.md)** documents every public function and class.
