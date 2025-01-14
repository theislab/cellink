#!/bin/sh
CMD="${SNAKEMAKE_CLUSTER_SIDECAR_VARS}${1}"
LOG="slurm-status.log"

if [ "$SNAKEMAKE_SLURM_DEBUG" = "1" ]; then
	{
    	echo "args: $@"
    	echo "executing ${CMD}"
	} >> $LOG

    eval $CMD | tee -a $LOG
else
    eval $CMD
fi
