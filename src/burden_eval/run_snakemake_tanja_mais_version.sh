#!/bin/bash
#SBATCH --job-name=comp_burdens          # Name of the job
#SBATCH --output=out_burden.txt        # File to write standard output
#SBATCH --error=error_burden.txt          # File to write standard error
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPUs per task
#SBATCH --mem=80GB                   # Memory per node (4 GB)
#SBATCH --partition=student_project        # Partition/queue to submit to
#SBATCH --gres=gpu:1


# sbatch slurm_burden_testing.sh

# activate conda env
source /data/nasif12/modules_if12/SL7/i12g/miniforge/24.9.0-0/etc/profile.d/conda.sh
conda activate mm_scgenetics

# run burden testing
echo "Running Snakemake"
snakemake --snakefile /data/nasif12/home_if12/l_mai/sc-genetics/src/burden_eval/Snakefile --cores 8