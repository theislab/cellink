# Single-cell Genetics (Cellink)

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/Single-cell Genetics (Cellink)/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/Single-cell Genetics (Cellink)

cellink enables genetic analyses on single-cell data

## Getting started

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/theislab/sc-genetics.git
git clone https://github.com/AIH-SGML/ratpy.git
uv venv
uv pip install -e sc-genetics -e ratpy

```

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [Mambaforge][].

There are several alternative options to install Single-cell Genetics (Cellink):

<!--
1) Install the latest release of `Single-cell Genetics (Cellink)` from [PyPI][]:

```bash
pip install Single-cell Genetics (Cellink)
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/theislab/sc-genetics.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/

[issue tracker]: https://github.com/theislab/Single-cell Genetics (Cellink)/issues
[tests]: https://github.com/theislab/sc-genetics/actions/workflows/test.yml
[documentation]: https://Single-cell Genetics (Cellink).readthedocs.io
[changelog]: https://Single-cell Genetics (Cellink).readthedocs.io/en/latest/changelog.html
[api documentation]: https://Single-cell Genetics (Cellink).readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/Single-cell Genetics (Cellink)
