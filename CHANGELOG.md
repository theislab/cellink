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
  syncing works, selecting subsets, aggregating, and saving/loading. No analysis, just the
  data structure itself
- `tl.external.build_known_cis_eqtls_from_tensorqtl`: build a known-cis-eQTL annotation
  (variant x gene, binary) from a completed TensorQTL nominal cis-scan, for use as a
  fine-mapping prior

### Fixed

- `DonorData.copy()` built a genuinely new object but always copied `_G`/`_C`
  regardless of whether they were views, unlike its previous behavior; reverted to
  only copying `_G`/`_C` when they're actually views (mutating `self` and returning
  it, as before)
- `io.read_zarr_dd` eagerly materialized `G`/`C` on load, unlike `io.read_pgen_zarr`
  (which already kept a dense `X` Dask-backed); it now uses the same lazy read path,
  so selecting a subset (e.g. one cell type of `C`) never touches the rest of the data
- `DonorData.write_zarr_dd` picked its own chunk shape for a dense `X` without regard
  to any chunking the input already had (confirmed: it disregarded a Dask array's own
  `.chunks` entirely), which for a genome-scale `X` could pick a shape badly misaligned
  with the data's on-disk layout and make the write far slower than necessary; `X` is
  now written separately with sane chunking (a Dask array's own chunks, or capped at
  4096 per axis for a plain array), overridable via `write_zarr_dd`'s new `x_chunks`
  argument; the same fix now also applies to any dense `layers` entry (previously only
  `X` was chunked sanely, so a dense layer the same shape as `X` (e.g. raw counts kept
  alongside a normalized `X`) still got the pathological chunking)
- `io.read_zarr_dd`'s lazy read only kept `X` Dask-backed, not a dense `layers` entry,
  so a dense layer was still eagerly materialized in full on load (and could exhaust
  memory for a genome-scale one); `layers` is now read lazily the same way `X` is
- `io._pgen._read_pvar` only mapped a `.pvar`'s columns to cellink's canonical
  `chrom`/`pos`/`snp_id`/`a0`/`a1` names when a `#CHROM` header line was present;
  a headerless `.pvar` now gets the same canonical mapping, assuming the standard
  PLINK2 column order, instead of being left with unnamed columns
- `io.from_sgkit_dataset` didn't cast `chrom`/`a0`/`a1` to `str`, unlike the pgen
  reader, which could leave a numeric `chrom` dtype that breaks a downstream string
  membership check (e.g. in tensorQTL)
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
- `io.to_plink` crashed (`TypeError: NDFrame.to_csv() got an unexpected keyword
  argument 'line_terminator'`) with `pandas-plink` 2.2.9 on a modern pandas, which
  renamed/removed that `to_csv` kwarg; bumped the minimum `pandas-plink` version to
  2.3.0, which uses the current `lineterminator` kwarg internally

### Known limitations

- Reading a sparse (`csr`/`csc`) `X` or layer back from a Zarr v3 store is slow:
  confirmed via two independent hand-built-matrix tests that plain
  `anndata.io.read_elem` alone (no cellink code involved) takes ~19s for a
  5M-nnz array that writes in ~0.2s, and `DonorData.write_zarr_dd`/`read_zarr_dd`
  add only ~9% on top of that baseline, so this is inherent to AnnData's
  sparse Zarr v3 read path in this environment, not a `DonorData`-specific
  issue. Not yet root-caused further, and not currently fixed, since no real
  data in this codebase's own pipelines uses a sparse Zarr layer today (dense
  `X`/layers only; the sparse `rare`-stratum genotype is read from PGEN
  directly, never persisted to Zarr). Revisit if/when a real sparse Zarr
  layer is added.

### Changed

- Expanded tutorial index and API reference to include the LIVI, sc-linker, MAGMA and
  cell-level LDSC tutorials/functions that existed but weren't documented
- Removed the stale cookiecutter `example.ipynb` placeholder notebook
