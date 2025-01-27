#!/bin/bash
#SBATCH --job-name=data_analysis          # Name of the job
#SBATCH --output=out_data_analysis.txt        # File to write standard output
#SBATCH --error=err_data_analysis.txt          # File to write standard error
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=1          # Number of CPUs per task
#SBATCH --mem=32GB                   # Memory per node (4 GB)


# sbatch slurm_burden_testing.sh

# run burden testing
echo "Running run_analysis.py"

python analysis.py -i /s/project/sys_gen_students/2024_2025/project04_rare_variant_sc/input_data/OneK1K_cohort_gene_expression_matrix_14_celltypes_w_gene_locations.h5ad.gz -o /s/project/sys_gen_students/2024_2025/project04_rare_variant_sc/output/data_plots