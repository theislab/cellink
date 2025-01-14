#!/bin/bash
# All the slurm arguments can be found here https://slurm.schedmd.com/sbatch.html
# https://snakemake.readthedocs.io/en/stable/executing/cli.html
# Maintainers:
#   Vangelis Theodorakis theodora@in.tum.de
#   Florian R. Hoelzlwimmer hoelzlwi@in.tum.de
#
# Fail the script if one command fails
set -e
# # enable debug mode
# set -x
# export SNAKEMAKE_SLURM_DEBUG=1

# limit core dumps to 50MB
#ulimit -c 102400

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ============================================================================
#
# 1) Make sure that you have snakemake installed in your $PATH
#
# 2) Specify the following arguments according to your own taste
#
# 3) Run ./run_slurm_jobs.sh
#


# Use the default snakemake command determined by your $PATH
# otherwise specify absolute path to snakemake binary
snakemake="snakemake"


# Change kinit path to use system kerberos instead of potential conda
# installed versions
kinit="/usr/bin/kinit"

# The name of the snakefile
snakefile="${SNAKEFILE:-Snakefile}"

# The number of snakemake cores
number_of_snakemake_cores="${N_CORES:-256}"
number_of_snakemake_jobs="${N_JOBS:-128}"
amount_of_snakemake_memory="${MEM_MB:-1000000}"
number_of_snakemake_gpus="${N_GPUS:-8}"

#### IMPORTANT!!!
# Make a environment variable for the project folder
# e.g. project_folder="/path/to/project/folder/"
project_folder="$(dirname "$snakefile")"
project_folder="$(realpath "$project_folder")"
project_name="$(basename "$project_folder")"

# Set the log folder path
logs="$project_folder/logs"

# Set the job name for the job that will be spawned
job_names="${project_name}-$(date +"%Y-%m-%d_%T")"
echo "Starting $job_names with up to:"
echo "- $number_of_snakemake_cores cores"
echo "- $number_of_snakemake_jobs jobs"
echo "- ${amount_of_snakemake_memory}MB of RAM"
echo "- $number_of_snakemake_gpus gpus"

snakemake_version="$($snakemake --version | sed -e 's/\([0-9]\+\).*/\1/')"
if [ "$snakemake_version" -gt "6" ]; then
	# snakemake supports the cluster-sidecar script
    cluster_status_script="--cluster-status=${SCRIPT_DIR}/slurm-status.sh"
    cluster_sidecar_script="--cluster-sidecar=${SCRIPT_DIR}/slurm-sidecar.py"
else
    # snakemake does not support cluster-sidecar script; use legacy polling instead
    cluster_status_script="--cluster-status=${SCRIPT_DIR}/slurm-status.py"
    cluster_sidecar_script=""
fi

# ============================================================================

# By default errors and outputs are printed in the same file
# so here we store the successfull outputs as .out files
# and the errors as .error files

output_files="$logs/%A-$job_names.out"

# Create the log folder if it does not exist
if [[ ! -e "$logs" ]]; then
    mkdir "$logs"
    echo "New logs folder created under $logs"
fi

# register cleanup function to stop still running snakemake jobs
function cleanup {
  echo "cancel still running jobs..."
  squeue -u $USER -o "%j,%i,%T,%B,%A,%N" | grep "^$job_names" | cut -f2 -d',' | xargs -r scancel
}

trap cleanup EXIT

## Run the snakemake file on the cluster

# requeue jobs in case of preemption
SBATCH_ARGS="${SBATCH_ARGS} --requeue"

if [ "${AUKS_ENABLED:-false}" = true ]; then
    # Fetch kerberos ticket that lasts for 7 days
    $kinit -r 7d

    # Auks argument caches the kerberos ticket for runs that last more than
    # one day (otherwise the jobs lose access to the filesystem)
    auks -a
    SBATCH_ARGS="${SBATCH_ARGS} --auks=done"
fi

$snakemake --keep-going \
  --default-resources ntasks=1 mem_mb=64000 gpu=0 \
  --cluster "sbatch $SBATCH_ARGS \
    --ntasks {resources.ntasks} \
    --cpus-per-task {threads} \
    --parsable \
    --mem {resources.mem_mb}M \
    --output $output_files \
    --job-name=$job_names-{rule} \
    --gres=gpu:{resources.gpu} \
  " \
  ${cluster_status_script} \
  ${cluster_sidecar_script} \
  --cores $number_of_snakemake_cores \
  -j $number_of_snakemake_jobs \
  --resources mem_mb=$amount_of_snakemake_memory gpu=$number_of_snakemake_gpus \
  --snakefile $snakefile "$@"
  # --verbose
  # --rerun-incomplete
