# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Basic tool, preprocessing and plotting functions
- LIVI donor-level representation learning, sc-linker gene programs, scPRS, gsMap and
  MAGMA wrappers under `cellink.tl.external`, now documented in the API reference
- Tests for `cellink.io.stream_pgen_to_zarr`/`read_pgen_zarr` and the `cellink-pgen` CLI
- Tests for previously untested pure functions (`utils.column_normalize`, `utils.gaussianize`,
  `tl.external.scores_to_gmt`/`scores_to_covar`, `tl.external.compute_escore`, `JointNMFWrapper`)
- New "DonorData basics" tutorial: building a `DonorData` from your own genotype/expression
  data (including loading genotypes via `read_sgkit_zarr`/`stream_pgen_to_zarr`), how donor
  syncing works, selecting subsets, aggregating, and saving/loading — no analysis, just the
  data structure itself

### Fixed

- `pp.log_transform` and `pp.normalize` no longer mutate the input `AnnData` when
  called with `inplace=False`
- `Skat` and `StructLMM` were unusable (missing `limix-core` imports); both now work,
  with the import done lazily and a clear error if `limix-core`/`limix-lmm` is missing
- `at.utils.davies_pvalue` no longer references an undefined `chiscore` import;
  `chiscore` is now imported lazily with an actionable install message
- `cellink.tl.external`'s `__all__` listed several names that were never imported
  (leftover from renames), which broke `from cellink.tl.external import *`
- `io.stream_pgen_to_zarr` never returned anything, contradicting its docstring; it now
  optionally returns the written `AnnData` via a new `return_adata` argument (default
  `False`, since this is normally a one-off conversion step)
- `tl.external.JointNMFWrapper` crashed on newer scipy versions (sparse matrix + scalar)
- Fixed a broken tutorial link and stale template placeholders in the README and
  contributing guide
- A stray unanchored `data` entry in `.gitignore` silently hid any new file added
  under `tests/data/` or `docs/tutorials/data/`; scoped it to the latter only

### Changed

- Expanded tutorial index and API reference to include the LIVI, sc-linker, MAGMA and
  cell-level LDSC tutorials/functions that existed but weren't documented
- Removed the stale cookiecutter `example.ipynb` placeholder notebook
