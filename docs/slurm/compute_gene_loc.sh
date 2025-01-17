#!/bin/bash
#SBATCH --job-name=burden_test          # Name of the job
#SBATCH --output=compute_gene_loc.txt        # File to write standard output
#SBATCH --error=compute_gene_loc_err.txt          # File to write standard error
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=1          # Number of CPUs per task
#SBATCH --mem=32GB                   # Memory per node (4 GB)

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mai.mai@tum.de,merit.back@tum.de,T.Pock@campus.lmu.de

# sbatch slurm_burden_testing.sh

# activate conda env
# source /data/nasif12/modules_if12/SL7/i12g/miniforge/24.9.0-0/etc/profile.d/conda.sh
# conda activate mzb_scgenetics

# run burden testing
echo "Running compute_gene_locations.py"

python compute_gene_locations.py