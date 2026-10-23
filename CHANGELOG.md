# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Changed

- `DonorData`, its readers and writers, the donor and cell filters in `pp` and `MILDataset` moved to the
  new `donordata` package and are re-exported by cellink. The donor level is now called `D` and `dd.G` is gone,
  files with a donor level `G` are read into `D` with a warning. It can also be a plain table of donor covariates with donor-aligned matrices in `dd.obsm` and
  `dd.obsp`, and an optional sample level `S` sits between donors and cells.
- The DonorData file format is now version 0.3.0. Files written as 0.1.0 or 0.2.0 are still read.
- The EHR tutorial moved to the DonorData documentation.
- `pp.cell_level_obs_filter`, `pp.donor_level_obs_filter` and `pp.donor_level_var_filter` now modify the
  object when `inplace=True`, which they silently did not before.
- `tl.external.run_mixmil` builds bags in the donor order of `G`, which previously could misalign bags
  and labels when donors were not sorted alphabetically.
- `muon` is no longer a dependency.

### Added

- Config arguments (`config_path` / `config_file`) across cellink accept the name of a
  config shipped with cellink, e.g. `tl.external.configure_ldsc_runner("ldsc_docker")`,
  `tl.run_vep(config_file="vep_offline")` or
  `resources.get_1000genomes_plink_files(config_path="1000genomes_grch38")`; a path to your
  own YAML works as before. A bare name refers to a shipped config before a file of that
  name in the working directory
- `config_path` on `tl.external.configure_livi_runner`, which the shipped `livi.yaml`
  already documented but which did not exist; arguments left unset are taken from the config
- Fixed `SCLINKER_ENHANCER_LINKS_GENOME_BUILD`: the Broad sc-linker Roadmap/ABC
  enhancer-gene links are GRCh37, not GRCh38. Declaring GRCh38 refused correct
  GRCh37 setups and, worse, let a GRCh38 panel pass the build check while
  intersecting hg19 enhancers against hg38 SNPs. Verified from the files: ABC
  `TargetGeneTSS` matches hg19 exactly for BACH2/CTLA4/FOXP3, and the Roadmap
  file's largest chr1 coordinate (249,240,000) exceeds GRCh38 chr1's length.
- `resources.get_eqtl_catalog_credible_sets` and `resources.get_eqtl_catalog_lbf`,
  exposing the eQTL Catalogue's SuSiE fine-mapping output (per-variant PIPs, and the
  per-variant log Bayes factors that `tl.coloc_susie` needs for the QTL side -- there
  was previously no way to obtain these through cellink)
- `region=` on `resources.get_eqtl_catalog_dataset_associations`, performing a remote
  tabix range query instead of downloading a whole dataset (a single dataset's
  summary statistics are ~1.4 GB)
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
- New scEPS tutorial: end-to-end walkthrough of the four [scEPS](https://github.com/Genentech/sceps)
  command-line steps (per-neighborhood statistics, approximately independent neighborhood
  blocks, cell-type aggregation, correlation with gene expression) on OneK1K, for rheumatoid
  arthritis. Follows the scEPS paper's design for a healthy cohort, where no case/control
  status exists to decompose: the phenotype is an RA polygenic risk score computed from the
  donors' own `dd.G` dosages, with age, sex and local cell proportion as covariates. Also
  covers gene-level statistics via `tl.external.run_magma_pipeline` and how to parallelize a
  real run
- `resources.get_pgs_catalog_score_file` is now exported from `cellink.resources` and listed
  in the API reference; it existed but could only be reached through the private module
- `resources.get_pgs_catalog_score_file(genome_build=...)` downloads the PGS Catalog's
  harmonized scoring file for that build (previously only the author-submitted file, in
  whatever build the authors used); it also takes `data_home`, and reuses an existing
  download unless `refresh=True`
- `data_home` on `tl.external.run_magma_pipeline` and `prepare_magma_inputs`, so the MAGMA
  reference files can live somewhere other than cellink's default data directory
- `tl.external.build_known_cis_eqtls_from_tensorqtl`: build a known-cis-eQTL annotation
  (variant x gene, binary) from a completed TensorQTL nominal cis-scan, for use as a
  fine-mapping prior

### Fixed

- Default configs only worked from particular working directories: the 1000 Genomes
  functions (`resources.get_1000genomes*`) defaulted to `./cellink/resources/config/...`,
  FAVOR and VEP to `../configs/...`, and MAGMA to `configs/magma.yaml`, so tutorials had
  to pass hand-built paths. Every default is now a shipped config resolved inside the
  installed package; the repo-level `configs/` directory moved into the package (`favor`
  and `magma` to `resources/config/`, `vep*` to `tl/external/config/`), and the tutorials
  no longer pass config paths
- `LDSCRunner` and `SAIGEQTLRunner` silently ignored a `config_path` that did not exist
  and ran with built-in defaults; they now raise, and read their defaults from the shipped
  `ldsc.yaml` / `saigeqtl.yaml` instead of a hard-coded copy of them
- Both eQTL Catalogue accessors were non-functional: `resources.get_eqtl_catalog_datasets`
  and `resources.get_eqtl_catalog_dataset_associations` targeted the retired REST API at
  `https://www.ebi.ac.uk/eqtl/api/v3`, which returns HTTP 410 for every endpoint and
  version. Both are rewritten against the current FTP/tabix distribution
  (https://www.ebi.ac.uk/eqtl/Data_access/). `max_pages` is still accepted but ignored
  with a warning, since the dataset index is no longer paginated
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
- `tl.external`'s MAGMA Entrez-to-Ensembl conversion (`gene_id_type="ensembl"`) assigned
  genes to Ensembl IDs on alternative haplotypes and patches, which never match expression
  data (e.g. C6orf10's MAGMA statistic landed only on two MHC-haplotype IDs), and gave one
  Entrez gene several Ensembl IDs through noisy cross-references (ACTB's statistic also
  under ACTG1's ID). The mapping is now restricted to the primary assembly, ties are broken
  by gene symbol, every Ensembl ID is unique, and genes such as HLA-DRB1 whose Ensembl ID
  carries several Entrez IDs are no longer dropped. Converted files are cached under a new
  name, so files written by the old mapping are not reused
- `tl.external.run_magma_pipeline`, `prepare_magma_inputs` and `download_magma_references`
  defaulted to `configs/magma.yaml`, a path that only exists in a source checkout; the
  config now ships inside the package and is the default

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
