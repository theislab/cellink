# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.0.2] - 2026-11-15

### Changed

- `tl.external.run_tensorqtl` adds no intercept covariate, which removed the first donor from every
  test

### Fixed

- `resources.get_gwas_catalog_study_summary_stats` and the eQTL Catalogue loaders give every column
  one type; a chromosome column mixed integers and strings (#158)

## [0.0.1] - 2026-11-15

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
- `jaxqtl` extra
- `sclinker` extra
- `n_jobs` on `tl.external.run_magma_pipeline`, `tl.external.genescores_to_annotations` and
  `tl.external.run_sclinker_heritability`
- `data_home` and `refresh` on `resources.get_gwas_catalog_study_summary_stats`
- `tl.external` exports `download_magma_references`, `get_gene_annotation` and `run_scprs`
- Tutorials: DonorData basics, scEPS
- API reference pages for sc-linker, scPRS, gsMap, MAGMA and cell-level LDSC

### Changed

- Python >= 3.12 is required; tested on Python 3.12 to 3.14. New minimum versions, e.g. numpy 2.2,
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
- `tl.external.run_jaxqtl` runs the jaxqtl 0.3 subcommands `cis`, `nominal` and `trans`, with one
  parameter per jaxqtl option and `compute-pcs` for expression PCs (`expression_pcs`);
  `tl.external.read_jaxqtl_results` reads the result file of a scan
- `tl.external.run_jaxqtl` fits its count models to summed counts with a log library-size offset;
  `libsize_key` takes the library sizes from a per-cell total over all genes (e.g. `nCount_RNA`)
- `tl.external.compute_joint_nmf_programs` couples the gene loadings; sc-linker bedgraphs keep the
  largest score of overlapping intervals
- `tl.external.compute_nmf_programs` follows sc-linker (Jagadeesh et al. 2022): it factorizes the
  log-normalized expression in `adata.X` (not the `counts` layer) with scikit-learn's NMF and random
  initialization (torchnmf only with `device`), fits on the highly variable genes when they are
  annotated, correlates every expressed gene with the factors, collapses factors whose contributions
  correlate above 0.8 (`collapse_threshold`) and writes gene scores (`{prefix}_genescores.csv`)
- `tl.external.estimate_ld_scores_from_donor_data` uses a 1000 kb window
- `tl.coloc_susie` defaults to `p12=5e-6`
- `at.acat_test` ignores NaN p-values, rejects non-positive weights and rescales the weights to sum to 1
- `pp.variant_qc` applies `hwe_pval_threshold`
- sc-linker gene coordinates have one span per Ensembl gene
- `resources.get_onek1k` lifts positions to hg19 correctly and names genotype PCs `gPCs_1` to
  `gPCs_N`
- `resources.get_dummy_onek1k` loads a new tutorial dataset (300 donors, at most 420 cells per
  donor)
- External tool commands run without a shell; BioMart queries use HTTPS with retries; downloads are
  atomic; the `tensorqtl` extra installs rpy2

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
- `tl.external.run_jaxqtl` writes gene coordinates as integers, reads results from `out` and returns
  the output paths with `read_results=False`
- `resources.liftover_gwas_summary_stats` with float chromosome labels and NA positions
- Per-query caches of `get_gwas_catalog_studies`, `get_gwas_catalog_genes`, `get_pgs_catalog_scores`
  and `get_1000genomes_merge_alleles`
- `tl.external.run_tensorqtl`: `maf_threshold` in the Python API, `read_results=False`, early error
  without R or qvalue
- Covariates with missing values, including donor sex, raise in `run_tensorqtl`, `run_jaxqtl` and
  `run_saigeqtl`; `run_saigeqtl` restricts cis tests to the gene window
- Singularity runs set `--pwd /data`
- LDSC helpers accept `ld_wind_kb` or `ld_wind_snp` alone and pass `maf_min`
- MAGMA input preparation: p-value clipping, allele order, prefixes with dots
- `tl.external.load_gsmap_results` reads only the requested trait
- `tl.external.compute_nmf_programs` gave genes without counts the correlations of other genes
- `tl.coloc_abf` input checks and very strong signals; `tl.coloc_susie` with no credible sets
- Edge cases in `tl.build_pca_informed_prior`, `tl.subset_gene` and `tl.subset_genomic_region`
- `cellink.pp` filters with `inplace=True` filter the input
- `DonorData.sel` aligns boolean Series by label
- `ml.DonorMILModel` with `cell_mask` and multi-class targets
- `pl.expression_by_genotype`, `pl.manhattan`, `pl.qq` and `pl.enrichment_scatter`
- `validate` checks MuData genotypes
- Path arguments of the MAGMA, gsMap, sc-linker and S-LDSC helpers expand `~`
- An interrupted extraction of a 1000 Genomes archive is redone instead of being used as complete;
  data extracted by an earlier version is extracted once more
- sc-linker's `bedgraph_to_snp_annotation` without bedtools uses the same coordinates as bedtools
  and is vectorized
- Tutorials: colocalization, scdrs_seismic, clumping_pruning, pseudobulk_eqtl, the jaxqtl/tensorQTL
  and SAIGE-QTL eQTL tutorials, cell_level_ldsc_analysis (genome builds, LDSC-SEG control gene set),
  sclinker (RA GWAS, gene names, AllCoding control), magma_analysis_tutorial (MAGMA build for the
  operating system, also macOS) and run_dataloader; every tutorial runs in under an hour

### Known limitations

- Reading a sparse `X` or layer from a Zarr v3 store is slow (AnnData's sparse Zarr v3 reader)
