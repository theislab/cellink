# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Config arguments (`config_path`/`config_file`) accept the name of a config shipped with cellink,
  e.g. `tl.external.configure_ldsc_runner("ldsc_docker")`
- `resources.get_eqtl_catalog_credible_sets` and `resources.get_eqtl_catalog_lbf`, and `region=` on
  `resources.get_eqtl_catalog_dataset_associations`
- `resources.get_pgs_catalog_score_file` in the public API, with `genome_build`, `data_home` and
  `refresh`
- `data_home` on `tl.external.run_magma_pipeline` and `tl.external.prepare_magma_inputs`
- `tl.external.build_known_cis_eqtls_from_tensorqtl`
- `io.to_pgen`, which writes PLINK 2 files (`.pgen`, `.pvar`, `.psam`), with dosages for non-integer
  values
- `lazy=True` on `io.read_h5_dd` and `io.read_dd`: dense `X` and layers are read from the file when
  computed, as with `io.read_zarr_dd`
- Tutorials: DonorData basics, scEPS
- API reference pages for sc-linker, scPRS, gsMap, MAGMA and cell-level LDSC

### Changed

- Python >= 3.12 is required; tested on Python 3.12 to 3.15. New minimum versions, e.g. numpy 2.2,
  pandas 2.3, scipy 1.15, anndata 0.12.2 and zarr 3.1
- `io.to_plink` writes PLINK 1 files without pandas-plink; `io.to_plink` and `io.to_pgen` read
  negative values of an integer genotype matrix as missing calls, as in int8 stores read with
  `backend="zarr"`
- `io.read_plink` reads `.bim` and `.fam` files separated by any whitespace, strips a trailing
  `.bed` from `path` and keeps chromosome code 0 as `"0"`
- `io.to_plink` writes ALT to `.bim` column 5 and `io.read_plink` counts that allele;
  `count_a1=False` reads files with REF in column 5
- `tl.external.run_tensorqtl` writes the genotypes with `io.to_pgen` and does not need plink2
- `io.read_zarr_dd` and `io.read_pgen_zarr` return missing genotype calls as NaN
- `io.from_sgkit_dataset` stores per-call fields as layers (`GENOTYPE_0`, `GENOTYPE_1`, ...) instead
  of in `uns`
- Install messages name the PyPI distribution, e.g. `pip install 'cellink-tools[pgen]'`
- Directly used dependencies are declared (numpy, scipy, h5py, matplotlib, seaborn, pyyaml, tqdm);
  `pynndescent<0.7`; the `datasets` extra requires `bio2zarr>=0.2`
- The `ldsc` extra does not include gsMap, which needs its own environment; the `scdrs` extra does
  not pin numpy<2
- The `scooby` extra does not include scooby, which is not on PyPI; the scooby functions show the
  command that installs it from GitHub (Python 3.12 only). The `tensorqtl` extra installs tensorqtl
  1.0.10, the version cellink's tensorQTL command line is written for, and only on Python < 3.14
- `DonorData` keeps the order of cells within each donor

### Removed

- LIVI wrappers, extra, config and tutorial: LIVI's association testing imports pandas-plink, which
  requires numpy < 2. They are kept on the branch `feature/livi`
- `pandas-plink` and `bed-reader` from the core dependencies
- The `example.ipynb` placeholder notebook

### Fixed

- `tl.external.run_scdrs`, `run_seismic` and `run_seismic_torch` decide whether to normalize from
  the values of `X` and do not modify the input
- Default configs resolve inside the installed package; `LDSCRunner` and `SAIGEQTLRunner` raise for
  a missing config file
- eQTL Catalogue accessors use the current FTP/tabix distribution
- `SCLINKER_ENHANCER_LINKS_GENOME_BUILD` is GRCh37
- `DonorData.copy()` copies `G` and `C` only when they are views
- `io.read_zarr_dd` reads `X` and dense layers lazily; `DonorData.write_zarr_dd` chunks them
  (`x_chunks`)
- Headerless `.pvar` files; `io.from_sgkit_dataset` reads `chrom`, `a0` and `a1` as strings
- `pp.log_transform` and `pp.normalize` with `inplace=False` do not modify the input
- `Skat`, `StructLMM` and `at.utils.davies_pvalue` import their dependencies lazily
- `tl.external.__all__` lists only existing names
- `io.stream_pgen_to_zarr(return_adata=True)` returns the AnnData
- `tl.external.JointNMFWrapper` with current SciPy
- MAGMA's Entrez-to-Ensembl conversion uses the primary assembly and one Ensembl ID per gene; the
  MAGMA config ships with the package
- README and contributing links; `.gitignore` hid files under `tests/data/`
- `io.to_plink` wrote the maternal ID as paternal ID, truncated dosages and failed on missing sex
  values; it rounds dosages with a warning, raises for values outside [0, 2] and writes unknown sex
  as 0
- `io.to_plink` writes sex codes for "male"/"female" labels and 0/1 coding (`donor_sex_map`);
  negative codes, such as -1 from `io.read_plink`, are unknown
- `io.read_plink` keeps quantitative `.fam` phenotypes, which were read as missing (-1)
- `io.stream_pgen_to_zarr` keeps missing calls missing in int8 and sparse stores, indexes samples by
  IID and checks that all input files list the same samples
- `io.read_bgen` always raised; `io.from_sgkit_dataset(X_field="GP")` crashed; the sgkit readers'
  docs listed `X_field` values that raise
- NumPy 2, pandas 3, zarr 3, AnnData 0.13, SciPy 1.18, transformers 5 and rpy2 3.6: VCF Zarr
  alleles, `io.stream_pgen_to_zarr`, `DonorData.write_zarr_dd`, the tensorQTL wrappers,
  `tl.external.calculate_ld`, `calculate_pcs`, scooby model loading, `run_scdrs(group_analysis=...)`
  and the colocalization and gsMap tutorials

### Known limitations

- Reading a sparse `X` or layer from a Zarr v3 store is slow (AnnData's sparse Zarr v3 reader)
