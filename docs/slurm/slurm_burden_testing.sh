#!/bin/bash
#SBATCH --job-name=burden_test          # Name of the job
#SBATCH --output=out_burden_testing.txt        # File to write standard output
#SBATCH --error=error_burden_testing.txt          # File to write standard error
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=1          # Number of CPUs per task
#SBATCH --mem=64GB                   # Memory per node (4 GB)
#SBATCH --partition=student_project        # Partition/queue to submit to
#SBATCH --gres=gpu:1

# sbatch slurm_burden_testing.sh

# activate conda env
source /data/nasif12/modules_if12/SL7/i12g/miniforge/24.9.0-0/etc/profile.d/conda.sh
#conda activate mm_scgenetics
conda activate mzb_scgenetics

# run burden testing
echo "Running my SLURM job..."
#python /data/nasif12/home_if12/l_mai/sc-genetics/docs/slurm/run_burden_test.py
python3 ../../src/burden_eval/run_burden_testing.py -c 22 -p "/data/ceph/hdd/project/node_09/sys_gen_students/2024_2025/project04_rare_variant_sc/input_data/filter_vcf_r08/chr22.dose.filtered.R2_0.8.vcz" -o  "/s/project/sys_gen_students/2024_2025/project04_rare_variant_sc/test_output/chr22_all_results_DNA_LM_and_MAF_100k.pkl"
