# Contributing guide

**cellink** connects donor-level genetic data (`GenoAnnData`, via dask) with single-cell
omics data (`AnnData`/`MuData`) through the `DonorData` container, and wraps a large
number of external genetics tools (PLINK, MAGMA, LDSC, TensorQTL, SAIGE-QTL, gsMap,
sc-linker, scDRS, ...) under `cellink.tl.external` so they can be driven directly from
`DonorData`. This document covers what's specific to contributing to cellink; for the
general git/PR workflow, scverse's [developer documentation][scanpy developer guide]
(this project started from the `cookiecutter-scverse` template) is a good reference.

[scanpy developer guide]: https://scanpy.readthedocs.io/en/latest/dev/index.html

## Installing dev dependencies

In addition to the packages needed to _use_ this package,
you need additional python packages to [run tests](#writing-tests) and [build the documentation](#docs-building).

:::::{tabs}
::::{group-tab} Hatch
The easiest way is to get familiar with [hatch environments][], with which these tasks are simply:

```bash
hatch test  # defined in the table [tool.hatch.envs.hatch-test] in pyproject.toml
hatch run docs:build  # defined in the table [tool.hatch.envs.docs]
```

::::

::::{group-tab} Pip
If you prefer managing environments manually, you can use `pip`:

```bash
cd cellink
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test,doc]"
```

::::
:::::

[hatch environments]: https://hatch.pypa.io/latest/tutorials/environment/basic-usage/

### Optional extras for `tl.external`

Most of `cellink.tl.external` wraps a specific tool behind its own extra
(`pgen`, `ml`, `mixmil`, `tensorqtl`, `scdrs`, `scprs`, `seismic_torch`, `rvat`, `ldsc`,
`datasets`; see `[project.optional-dependencies]` in `pyproject.toml`), and each import
is lazy: `import cellink` never requires any of them, and calling a function whose
dependency is missing raises a clear `ImportError` telling you which extra to install.
When you're touching one of these wrappers, install just the extra(s) you need, e.g.
`pip install -e ".[pgen,test]"`.

A few (MAGMA, LDSC, PLINK/PLINK2, TensorQTL's CLI mode, SAIGE-QTL) additionally shell
out to an external binary that isn't distributed on PyPI/conda at all — tests exercising
those either skip via `pytest.importorskip`/an explicit binary check, or aren't run in
CI and are only expected to pass in an environment where the tool is installed
separately. Don't assume the base test env in CI covers every `tl.external` module.

## Code-style

This package uses [pre-commit][] to enforce consistent code-styles.
On every commit, pre-commit checks will either automatically fix issues with the code, or raise an error message.

To enable pre-commit locally, simply run

```bash
pre-commit install
```

in the root of the repository.
Pre-commit will automatically download all dependencies when it is run for the first time.

Alternatively, you can rely on the [pre-commit.ci][] service enabled on GitHub.
If you didn't run `pre-commit` before pushing changes to GitHub it will automatically commit fixes to your pull request, or show an error message.

If pre-commit.ci added a commit on a branch you still have been working on locally, simply use

```bash
git pull --rebase
```

to integrate the changes into yours.
While the [pre-commit.ci][] is useful, we strongly encourage installing and running pre-commit locally first to understand its usage.

Finally, most editors have an _autoformat on save_ feature.
Consider enabling this option for [ruff][ruff-editors] and [prettier][prettier-editors].

[pre-commit]: https://pre-commit.com/
[pre-commit.ci]: https://pre-commit.ci/
[ruff-editors]: https://docs.astral.sh/ruff/integrations/
[prettier-editors]: https://prettier.io/docs/en/editors.html

(writing-tests)=

## Writing tests

This package uses [pytest][] for automated testing. Please add a test for every
function you add or change — see scanpy's {doc}`scanpy:dev/testing` guide for general
pytest conventions. Small synthetic-data tests (`tests/conftest.py` has `adata`/`gdata`
fixtures built from `cellink._core.dummy_data`) are preferred over ones that depend on
`tests/data/*` fixtures or network access; add a real assertion, not just a call that
exercises the code path without checking its result.

Use the `slow` marker (`@pytest.mark.slow`, deselect with `pytest -m "not slow"`) for
tests that read from `tests/data/`, hit the network, or otherwise take non-trivial time.
For functions gated behind an optional extra, guard the test with
`pytest.importorskip("<package>")` rather than skipping the whole module, so the test
still runs wherever that extra is installed.

Most IDEs integrate with pytest and provide a GUI to run tests.
Just point yours to one of the environments returned by

```bash
hatch env create hatch-test  # create test environments for all supported versions
hatch env find hatch-test  # list all possible test environment paths
```

Alternatively, you can run all tests from the command line by executing

:::::{tabs}
::::{group-tab} Hatch

```bash
hatch test  # test with the highest supported Python version
# or
hatch test --all  # test with all supported Python versions
```

::::

::::{group-tab} Pip

```bash
source .venv/bin/activate
pytest
```

::::
:::::

in the root of the repository.

[pytest]: https://docs.pytest.org/

### Continuous integration

The [`Test`](https://github.com/theislab/cellink/actions/workflows/test.yaml) workflow
runs the test suite against the minimum and maximum supported Python version (3.10 and
3.12) on every push/PR to `main`, and again on a schedule twice a month so dependency
drift gets caught even without a PR. It only installs the base `[dev,test]` extras, so
anything gated behind an optional extra (see above) is skipped there unless its
lazy-import guard reports it cleanly. The
[`Check Build`](https://github.com/theislab/cellink/actions/workflows/build.yaml)
workflow builds the package and runs `twine check --strict` on it.

## Publishing a release

### Updating the version number

Before making a release, you need to update the version number in the `pyproject.toml` file.
Please adhere to [Semantic Versioning][semver], in brief

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1. MAJOR version when you make incompatible API changes,
> 2. MINOR version when you add functionality in a backwards compatible manner, and
> 3. PATCH version when you make backwards compatible bug fixes.
>
> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

Once you are done, commit and push your changes and navigate to the "Releases" page of this project on GitHub.
Specify `vX.X.X` as a tag name and create a release.
For more information, see [managing GitHub releases][].
This will automatically create a git tag and trigger a Github workflow that creates a release on [PyPI][].

[semver]: https://semver.org/
[managing GitHub releases]: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
[pypi]: https://pypi.org/

## Writing documentation

Please write documentation for new or changed features and use-cases.
This project uses [sphinx][] with the following features:

- The [myst][] extension allows to write documentation in markdown/Markedly Structured Text
- [Numpy-style docstrings][numpydoc] (through the [napoloen][numpydoc-napoleon] extension).
- Jupyter notebooks as tutorials through [myst-nb][] (See [Tutorials with myst-nb](#tutorials-with-myst-nb-and-jupyter-notebooks))
- [sphinx-autodoc-typehints][], to automatically reference annotated input and output types
- Citations (like {cite:p}`Virshup_2023`) can be included with [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/)

See scanpy’s {doc}`scanpy:dev/documentation` for more information on how to write your own.

### Keeping the API reference in sync

Public functions/classes are exported from each submodule's `__init__.py` (and listed in
its `__all__`), and separately listed again in the matching `docs/api/*.md` page
(`donordata.md`, `pp.md`, `io.md`, `tl.md`, `tl_external.md`, `pl.md`, `ml.md`, `at.md`,
`utils.md`, `resources.md`, `cli.md`). These are two separate, manually-kept-in-sync
lists — nothing enforces that they match. When you add, rename, or remove a public
function:

1. Add it to the submodule's `__init__.py` import and `__all__` (or remove it from both).
2. Add/update the corresponding `docs/api/*.md` entry.
3. If it's a new tutorial notebook, add it to `docs/tutorials/index.md`'s toctree — a
   notebook that exists under `docs/tutorials/` but isn't listed there won't show up in
   the rendered docs navigation.

`ruff`'s `RUF100`/`F822` checks catch a stale `__all__` (a name listed that no longer
exists), but nothing currently catches the reverse (a public name that exists but was
never added to `__all__`/the API docs) or a tutorial notebook missing from the toctree —
worth double-checking by eye.

[sphinx]: https://www.sphinx-doc.org/en/master/
[myst]: https://myst-parser.readthedocs.io/en/latest/intro.html
[myst-nb]: https://myst-nb.readthedocs.io/en/latest/
[numpydoc-napoleon]: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[sphinx-autodoc-typehints]: https://github.com/tox-dev/sphinx-autodoc-typehints

### Tutorials with myst-nb and jupyter notebooks

The documentation is set-up to render jupyter notebooks stored in the `docs/tutorials` directory using [myst-nb][].
Currently, only notebooks in `.ipynb` format are supported that will be included with both their input and output cells.
It is your responsibility to update and re-run the notebook whenever necessary — notebooks aren't executed as
part of CI, so a stale output cell showing an old error/result won't be caught automatically.

If you are interested in automatically running notebooks as part of the continuous integration,
please check out [this feature request][issue-render-notebooks] in the `cookiecutter-scverse` repository.

[issue-render-notebooks]: https://github.com/scverse/cookiecutter-scverse/issues/40

#### Hints

- If you refer to objects from other packages, please add an entry to `intersphinx_mapping` in `docs/conf.py`.
  Only if you do so can sphinx automatically create a link to the external documentation.
- If building the documentation fails because of a missing link that is outside your control,
  you can add an entry to the `nitpick_ignore` list in `docs/conf.py`

(docs-building)=

#### Building the docs locally

:::::{tabs}
::::{group-tab} Hatch

```bash
hatch docs:build
hatch docs:open
```

::::

::::{group-tab} Pip

```bash
source .venv/bin/activate
cd docs
make html
(xdg-)open _build/html/index.html
```

::::
:::::
