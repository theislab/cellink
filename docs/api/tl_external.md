## External tools `tl.external`

Wrappers around external genetics tools. Most of these shell out to a command-line
tool that has to be installed separately (e.g. PLINK, MAGMA, LDSC, TensorQTL,
SAIGE-QTL) — see each tool's own documentation for installation instructions.

### eQTL mapping (jaxQTL / TensorQTL)

```{eval-rst}
.. module:: cellink.tl.external
.. currentmodule:: cellink

.. autosummary::
   :toctree: ../generated/

   tl.external.run_jaxqtl
   tl.external.read_jaxqtl_results
   tl.external.run_tensorqtl
   tl.external.read_tensorqtl_results
```

### SAIGE-QTL

```{eval-rst}
.. autosummary::
   :toctree: ../generated/

   tl.external.configure_saigeqtl_runner
   tl.external.get_saigeqtl_runner
   tl.external.make_group_file
   tl.external.run_saigeqtl
   tl.external.read_saigeqtl_results
```

### MixMIL

```{eval-rst}
.. autosummary::
   :toctree: ../generated/

   tl.external.run_mixmil
```

### LD and principal components

```{eval-rst}
.. autosummary::
   :toctree: ../generated/

   tl.external.calculate_ld
   tl.external.calculate_pcs
```

### LD score regression (S-LDSC)

```{eval-rst}
.. autosummary::
   :toctree: ../generated/

   tl.external.configure_ldsc_runner
   tl.external.munge_sumstats
   tl.external.make_annot_from_bimfile
   tl.external.make_annot_from_donor_data
   tl.external.estimate_ld_scores_from_bimfile
   tl.external.estimate_ld_scores_from_donor_data
   tl.external.compute_ld_scores_with_annotations_from_bimfile
   tl.external.compute_ld_scores_with_annotations_from_donor_data
   tl.external.estimate_heritability
   tl.external.estimate_celltype_specific_heritability
   tl.external.estimate_genetic_correlation
   tl.external.generate_gene_coord_file
   tl.external.generate_sldsc_genesets
   tl.external.get_magma_gene_loc
   tl.external.preprocess_for_sldsc
```

### MAGMA

```{eval-rst}
.. autosummary::
   :toctree: ../generated/

   tl.external.run_magma_pipeline
   tl.external.run_magma_annotate
   tl.external.run_magma_gene_analysis
   tl.external.run_magma_gsa
   tl.external.run_magma_gpa
   tl.external.genesets_dir_to_entrez_gmt
   tl.external.load_ensembl_to_entrez_map
   tl.external.scores_to_gmt
   tl.external.scores_to_covar
```

### gsMap

```{eval-rst}
.. autosummary::
   :toctree: ../generated/

   tl.external.format_gsmap_sumstats
   tl.external.load_gsmap_results
```

### sc-linker gene programs and heritability

```{eval-rst}
.. autosummary::
   :toctree: ../generated/

   tl.external.compute_celltype_programs
   tl.external.compute_diseaseprogression_programs
   tl.external.compute_nmf_programs
   tl.external.compute_joint_nmf_programs
   tl.external.JointNMFWrapper
   tl.external.compute_escore
   tl.external.run_sclinker_heritability
   tl.external.compute_ld_scores_for_sclinker
   tl.external.load_sclinker_heritability_results
   tl.external.download_sclinker_references
   tl.external.download_sclinker_enhancer_links
   tl.external.load_roadmap_links
   tl.external.load_abc_links
   tl.external.load_gene_annotation
   tl.external.genescores_to_abc_road_bedgraph
   tl.external.genescores_to_100kb_bedgraph
   tl.external.genescores_to_annotations
   tl.external.bedgraph_to_snp_annotation
   tl.external.check_and_patch_ldsc_parse_bug
```

### scDRS and seismic (GWAS x single-cell disease relevance)

```{eval-rst}
.. autosummary::
   :toctree: ../generated/

   tl.external.run_scdrs
   tl.external.run_seismic
   tl.external.run_seismic_torch
   tl.external.SparseScore
   tl.external.RegressionNLL
```

### scPRS

```{eval-rst}
.. autosummary::
   :toctree: ../generated/

   tl.external.prepare_scprs_data
   tl.external.get_plink_commands_per_cell
   tl.external.write_slurm_array_job
   tl.external.get_disease_relevant_cells
   tl.external.run_scprs_pipeline
```

### LIVI

Donor-level representation learning on paired genotype/expression data. See the
{doc}`../tutorials/livi` tutorial for a full walkthrough.

```{eval-rst}
.. autosummary::
   :toctree: ../generated/

   tl.external.LIVIRunner
   tl.external.configure_livi_runner
   tl.external.get_livi_runner
   tl.external.train_livi
   tl.external.infer_livi
   tl.external.run_livi_association_testing
   tl.external.save_livi_results
   tl.external.load_livi_results
```

`train_livi_annbatch`, `build_annbatch_collection`, `read_g_from_dd_store`, `CisGenotype`,
`LIVICisBatchAdapter` and `AnnbatchLIVIDataModule` provide an `annbatch`-streamed training
path for datasets too large to fit in memory. They require the extra `torch`,
`pytorch_lightning` and `annbatch` dependencies.
