# Single-cell Genetics Package (Cellink)

<!-- TODO comment back in once package is public -->
<!-- [![Tests][badge-tests]][tests]

<!-- [badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/Single-cell Genetics (Cellink)/test.yaml?branch=main -->
<!-- [![Documentation][badge-docs]][documentation]  -->
<!-- [badge-docs]: https://cellink-docs.readthedocs.io/  -->


Welcome to the official documentation for **cellink**—the toolkit designed to bridge the gap between single-cell data and individual-level genetic analysis.

## Motivation

Integrating genetic data with cellular heterogeneity is crucial for advancing personalized medicine. **cellink** provides the missing framework for efficiently handling and analyzing genetic variation alongside complex single-cell omics data at scale.

## ✨ Key Features & Structure

**cellink** introduces the `DonorData` class, unifying individual-level and single-cell data. It extends standard formats (AnnData, MuData) with GenoAnnData for efficient genotype (via dask) and phenotype (via ehrapy) handling.

```{eval-rst}
.. image:: _static/img/schematic_figure.png
    :width: 750px
    :alt: Data structure schematic
```

- **Donor-level Data (G):** `GenoAnnData`, Stores individual level data such as genotypes.
- **Cell-level Data (C):** `AnnData`/ `MuData`, Stores single-cell omics data such as gene expression.

Crucially, **`DonorData`** ensures that genetic data and single-cell modalities remain **synchronized**, preserving their donor-cell pairing even through complex filtering operations (e.g., selecting specific cell types or patient subsets).

### 2. Comprehensive Toolkit

**cellink** offers a streamlined suite of tools for the entire analysis workflow:

- **[Variant Preprocessing & Annotation](https://cellink-docs.readthedocs.io/en/latest/tutorials/explore_annotations.html):** Tools for quality control, annotation (VCF export/import), and selection of genetic variants.
- **Specialized Downstream Analysis:** Easily perform complex genetic analyses on single-cell expression data, including:
    - [eQTL mapping](https://cellink-docs.readthedocs.io/en/latest/tutorials/pseudobulk_eqtl.html).
          <!-- * Colocalization analysis with established disease loci. -->
    - [Rare variant association studies]([tutorials/burden_testing.ipynb](https://cellink-docs.readthedocs.io/en/latest/tutorials/burden_testing.html)).
- **Interoperability:** **cellink** enhances standard workflows through data exports compatible with common genetic analysis tools, e.g., for [eQTL analysis with jaxqtl or tensorqtl](https://cellink-docs.readthedocs.io/en/latest/tutorials/pseudobulk_eqtl_jaxqtl_tensorqtl.html) and includes built-in [dataloaders for deep learning](https://cellink-docs.readthedocs.io/en/latest/tutorials/run_dataloader.html).

## 🚀 Getting Started

- Check out the **[Tutorials](https://cellink-docs.readthedocs.io/en/latest/tutorials/index.html)** section for step-by-step guides on analysis workflows.
- Explore the **[API Reference](https://cellink-docs.readthedocs.io/en/latest/tutorials/api/index.html)** for detailed documentation.

Install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/theislab/sc-genetics.git@main
```

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
[issue tracker]: https://github.com/theislab/sc-genetics/issues

<!-- [tests]: https://github.com/theislab/sc-genetics/actions/workflows/test.yml
[documentation]: https://Single-cell Genetics (Cellink).readthedocs.io
[changelog]: https://Single-cell Genetics (Cellink).readthedocs.io/en/latest/changelog.html
[api documentation]: https://Single-cell Genetics (Cellink).readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/Single-cell Genetics (Cellink) -->
