<!-- logo-image-start -->

<p align="center">
  <img src="https://raw.githubusercontent.com/theislab/cellink/main/docs/_static/img/cellink-logo.svg" alt="cellink" width="360">
</p>

<br>

<!-- logo-image-end -->

[![Build](https://github.com/theislab/cellink/actions/workflows/build.yaml/badge.svg)](https://github.com/theislab/cellink/actions/workflows/build.yaml)
[![License](https://img.shields.io/github/license/theislab/cellink)](https://opensource.org/licenses/Apache-2.0)
[![Read the Docs](https://img.shields.io/readthedocs/cellink/latest.svg?label=Read%20the%20Docs)](https://cellink-docs.readthedocs.io/)
[![Test](https://github.com/theislab/cellink/actions/workflows/test.yaml/badge.svg)](https://github.com/theislab/cellink/actions/workflows/test.yaml)
[![PyPI](https://img.shields.io/pypi/v/cellink-tools.svg)](https://pypi.org/project/cellink-tools)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# cellink bridges donor genotypes and single-cell profiles for end-to-end single-cell genetics

cellink brings single-cell data together with the genomes of the donors the cells come from. It reads genotypes from
VCF, PLINK, BGEN and PGEN files, keeps them paired with the cells of the same donors through every selection, and
runs genetic analyses on both. Established tools such as PLINK, tensorQTL, SAIGE-QTL, LDSC and MAGMA run directly on
your data, and cellink takes care of the file formats and conventions each of them expects.

<!-- schematic-image-start -->

![cellink overview](https://raw.githubusercontent.com/theislab/cellink/main/docs/_static/img/schematic_figure.png)

<!-- schematic-image-end -->

## Data

cellink works on [`HiData`](https://hidata.readthedocs.io) objects, which link the cells of a cohort to their
donors. In cellink the group level `hd.G` holds the genotypes, one row per donor and one column per variant, and
`hd.C` holds the cells as an AnnData or MuData. Selecting donors, variants or cells keeps both levels aligned, so
genotypes and expression always describe the same people.

```python
import cellink as cl

genotypes = cl.io.read_plink("cohort")
hd = cl.HiData(G=genotypes, C=adata, group_id="individual")
hd = hd[..., hd.C.obs["cell_type"] == "CD4 T", :]
cl.io.write_hd(hd, "cohort.hd.zarr")
```

`HiData` is imported from the [HiData](https://github.com/theislab/hidata) package and is installed with cellink. The
[HiData documentation](https://hidata.readthedocs.io) explains how these objects are built and stored, and
how to add a subgroup level, such as samples, or the
[electronic health records](https://hidata.readthedocs.io/en/latest/tutorials/ehr_cohort.html) of the donors.

## Analyses

**Association testing**
- [eQTL mapping on pseudobulk with jaxQTL or tensorQTL](https://cellink-docs.readthedocs.io/en/latest/tutorials/pseudobulk_eqtl_jaxqtl_tensorqtl.html)
- [eQTL mapping on single cells with SAIGE-QTL](https://cellink-docs.readthedocs.io/en/latest/tutorials/single_cell_eqtl_saigeqtl.html)
- [A simple pseudobulk eQTL mapping](https://cellink-docs.readthedocs.io/en/latest/tutorials/pseudobulk_eqtl.html)
- [Rare variant association tests](https://cellink-docs.readthedocs.io/en/latest/tutorials/burden_testing.html)
- [Clumping and pruning](https://cellink-docs.readthedocs.io/en/latest/tutorials/clumping_pruning.html)
- [Colocalization](https://cellink-docs.readthedocs.io/en/latest/tutorials/colocalization.html)
- [Writing association tests as formula strings](https://cellink-docs.readthedocs.io/en/latest/tutorials/formula_resolver.html)

**Heritability, gene programs and GWAS integration**
- [LD score regression for cell types](https://cellink-docs.readthedocs.io/en/latest/tutorials/cell_level_ldsc_analysis.html)
- [Gene program scoring with sc-linker](https://cellink-docs.readthedocs.io/en/latest/tutorials/sclinker.html)
- [Gene set enrichment with MAGMA](https://cellink-docs.readthedocs.io/en/latest/tutorials/magma_analysis_tutorial.html)
- [Disease relevance of single cells from GWAS with scDRS and seismic](https://cellink-docs.readthedocs.io/en/latest/tutorials/scdrs_seismic.html)
- [Cell neighborhoods associated with disease with scEPS](https://cellink-docs.readthedocs.io/en/latest/tutorials/sceps.html)
- [GWAS on spatial data with gsMap](https://cellink-docs.readthedocs.io/en/latest/tutorials/gsmap.html)

**Deep learning**
- [Predicting single cell coverage from sequence and scoring variants with scooby](https://cellink-docs.readthedocs.io/en/latest/tutorials/scooby.html)
- [Dataloaders and multiple instance learning over the cells of each donor](https://hidata.readthedocs.io/en/latest/tutorials/mil_model.html), with `hidata.ml` from HiData

**Variants**
- [Quality control, annotation and selection of genetic variants](https://cellink-docs.readthedocs.io/en/latest/tutorials/explore_annotations.html)

## Getting started

Install cellink from PyPI, where it is published as `cellink-tools`. This also installs HiData:

```bash
pip install cellink-tools
```

To install the development version from GitHub instead:

```bash
pip install git+https://github.com/theislab/cellink.git@main
```

The [installation guide](https://cellink-docs.readthedocs.io/en/latest/installation.html) lists the optional extras
for the external tools. If you are new to cellink, start with
[HiData basics](https://cellink-docs.readthedocs.io/en/latest/tutorials/hidata_basics.html), which reads
genotype files into an object, subsets it and saves it without running any analysis.

## Documentation

| | |
|---|---|
| [Installation guide](https://cellink-docs.readthedocs.io/en/latest/installation.html) | Requirements and the optional extras (`pip install 'cellink-tools[extra-name]'`) |
| [Tutorials](https://cellink-docs.readthedocs.io/en/latest/tutorials/index.html) | Step by step guides for the analyses above |
| [API reference](https://cellink-docs.readthedocs.io/en/latest/api/index.html) | Every public function and class, organized by module |
| [HiData documentation](https://hidata.readthedocs.io) | The `HiData` object, its subgroup level and its file format |
| [Contributing guide](https://cellink-docs.readthedocs.io/en/latest/contributing.html) | Development setup, tests and the pull request workflow |
| [Changelog](https://cellink-docs.readthedocs.io/en/latest/changelog.html) | What changed in each release |

## Contact

<!-- For questions and help requests, you can reach out in the [scverse discourse][]. -->

If you found a bug, please use the [issue tracker](https://github.com/theislab/cellink/issues).

## Release notes

t.b.a

<!-- See the [changelog][]. -->

## Citation

> t.b.a

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/cellink/issues
