#!/usr/bin/env python3
import re
import subprocess as sp
import shlex
import sys
import time
import logging
import argparse

logger = logging.getLogger("__name__")


parser = argparse.ArgumentParser(description='Get the status of some slurm job.')
parser.add_argument(
    '--status_attempts',
    action='store',
    dest="status_attempts",
    type=int,
    default=20,
    help='number of attenpts to obtain cluster status'
)
parser.add_argument(
    '--cluster',
    action='store',
    dest="cluster",
    default="",
    help='cluster name'
)

parser.add_argument(
    '--debug',
    action='store_true',
    dest="debug",
    help='Change log level to debug'
)

parser.add_argument(
    'jobid',
    action='store',
    help='Slurm job id'
)

args = parser.parse_args()

if args.debug is True:
    loglevel = logging.DEBUG
else:
    loglevel = logging.WARNING

JOBID = args.jobid
STATUS_ATTEMPTS = args.status_attempts
if args.cluster != "":
    cluster = f"--cluster={args.cluster}"
else:
    cluster = ""

# setup logging
logging.basicConfig(
    level=loglevel,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("slurm-status.log"),
        # logging.StreamHandler()
    ]
)


def run_scontrol(cluster, jobid):
    sctrl_res = sp.check_output(
        shlex.split(f"scontrol {cluster} -o show job {jobid}")
    )
    m = re.search(r"JobState=(\w+)", sctrl_res.decode())
    res = {jobid: m.group(1)}

    if res[jobid] == "PREEMPTED":
        m = re.search(r"Requeue=(\w+)", sctrl_res.decode())
        requeueable = m.group(1)

        if requeueable == "1":
            res[jobid] = "REQUEUED"

    return res


def run_sacct(cluster, jobid):
    sacct_res = sp.check_output(shlex.split("sacct -P -b -j {} -n".format(jobid)))
    res = {x.split("|")[0]: x.split("|")[1] for x in sacct_res.decode().strip().split("\n")}
    return res


def get_state(jobid, status_attempts):
    for i in range(status_attempts):
        try:
            res = run_sacct(cluster, jobid)

            if res[jobid] == "PREEMPTED":
                # run scontrol to check whether the job will be requeued
                res = run_scontrol(cluster, jobid)

            break
        except sp.CalledProcessError as e:
            logger.error("sacct process error")
            logger.error(e)
        except IndexError as e:
            pass
    
        # Try getting job with scontrol instead in case sacct is misconfigured
        try:
            res = run_scontrol(cluster, jobid)
            break
        except sp.CalledProcessError as e:
            logger.error("scontrol process error")
            logger.error(e)
            if i >= status_attempts - 1:
                print("failed")
                exit(0)
            else:
                time.sleep(1)
    status = res[jobid]

    return status


def decide_return_value(jobid, status_attempts, preempt_retry=True, default="running"):
    status = get_state(jobid, status_attempts)
    logger.info("%s: '%s'", jobid, status)

    if status.startswith("CANCELLED") or status in [
        "BOOT_FAIL",
        "OUT_OF_MEMORY",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "TIMEOUT",
        "PREEMPTED",
    ]:
        logger.warning(f"Job '{jobid}' failed: '{status}'")
        return "failed"
    # elif status == "PREEMPTED":
    #     if preempt_retry:
    #         # wait for 10 seconds and then try again to see if it got requeued
    #         from time import sleep
    #         sleep(10)

    #         return decide_return_value(jobid, status_attempts, preempt_retry=False)
    #     else:
    #         logger.warning(f"Job '{jobid}' failed after preemption: '{status}'")
    #         return "failed"
    elif status in [
        "PENDING",
        "RUNNING",
        "REQUEUED",
        "COMPLETING",
        "SUSPENDED", # Unclear whether SUSPENDED should be treated as running or failed
    ]:
        return "running"
    elif status == "COMPLETED":
        return "success"
    else:
        logger.warning(f"Unkonwn job state of job '{jobid}': '{status}'")
        return default


status = decide_return_value(JOBID, STATUS_ATTEMPTS)

print(status)

