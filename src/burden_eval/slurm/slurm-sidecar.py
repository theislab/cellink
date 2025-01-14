#!/usr/bin/env python3
"""Run a Snakemake v7+ sidecar process for Slurm

This is a heavily modified version of the official Snakemake Slurm profile:
    https://github.com/Snakemake-Profiles/slurm/blob/master/%7B%7Bcookiecutter.profile_name%7D%7D/slurm-sidecar.py

---
MIT License

Copyright (c) 2017 Snakemake-Profiles

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import http.server
import json
import logging
import os
import subprocess
import sys
import signal
import time
import threading
import uuid
import argparse
import textwrap

# from CookieCutter import CookieCutter

parser = argparse.ArgumentParser(description=textwrap.dedent("""
    Run a Snakemake v7+ sidecar process for Slurm.
    This sidecar process will poll `squeue --user [user] --format='%i,%T'` 
    every 5 seconds by default.

    The first output line will be a cURL command that can be used to query job states in the following manner:
    `curl -s [...] http://localhost:<port>/job/status/$JOBID_OF_INTEREST`
"""))
parser.add_argument(
    "--cluster",
    action="store",
    dest="cluster",
    default=os.environ.get("SNAKEMAKE_CLUSTER", ""),
    help="cluster name",
)
parser.add_argument(
    "--user",
    action="store_true",
    dest="user",
    default=os.environ.get("USER", ""),
    help="SLURM user that submits the jobs",
)
parser.add_argument(
    "--json-conn",
    action="store_true",
    dest="json_conn",
    default=bool(int(os.environ.get("SNAKEMAKE_CLUSTER_SIDECAR_JSON_CONN", "0"))),
    help="Print the connection details as JSON dictionary instead of cURL command",
)
parser.add_argument(
    "--squeue_cmd",
    action="store",
    dest="squeue_cmd",
    default=os.environ.get("SNAKEMAKE_SLURM_SQUEUE_CMD", "squeue"),
    help="Command to call when calling squeue",
)
parser.add_argument(
    "--squeue_wait",
    action="store",
    dest="squeue_wait",
    type=int,
    default=int(os.environ.get("SNAKEMAKE_SLURM_SQUEUE_WAIT", "5")),
    help=textwrap.dedent("""Number of seconds to wait between ``squeue`` calls. 
        Note that you have to adjust the value to fit to your ``MinJobAge`` Slurm configuration.
        Jobs remain at least ``MinJobAge`` seconds known to the Slurm controller (default of 300 seconds).
        If you query ``squeue`` every 60 seconds then this is plenty and you will observe all relevant job status states as they are relevant for Snakemake.
    """),
)
parser.add_argument(
    "--log_request",
    action="store_true",
    dest="log_requests",
    default=bool(int(os.environ.get("SNAKEMAKE_SLURM_LOG_REQUESTS", "0"))),
    help="Enables HTTP request logging in sidecar",
)
parser.add_argument(
    "--debug",
    action="store_true",
    dest="debug",
    default=bool(int(os.environ.get("SNAKEMAKE_SLURM_DEBUG", "0"))),
    help="Change log level to debug",
)

args = parser.parse_args()


CLUSTER = args.cluster
if CLUSTER == "":
    CLUSTER = None

USER = args.user
if USER == "":
    USER = None

USE_CURL = not args.json_conn

LOG_REQUESTS = args.log_requests
SQUEUE_CMD = args.squeue_cmd
SQUEUE_WAIT = args.squeue_wait

logger = logging.getLogger(__name__)
#: Enables debug messages for slurm sidecar.
if args.debug is True:
    loglevel = logging.DEBUG
else:
    loglevel = logging.WARNING

# setup logging
logging.basicConfig(
    level=loglevel,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("slurm-sidecar.log"),
        # logging.StreamHandler()
    ],
)


class PollSqueueThread(threading.Thread):
    """Thread that polls ``squeue`` until stopped by ``stop()``"""

    def __init__(
        self,
        squeue_wait,
        squeue_cmd,
        squeue_timeout=2,
        sleep_time=1,
        max_tries=3,
        cluster=None,
        user=None,
        *args,
        **kwargs,
    ):
        super().__init__(target=self._work, *args, **kwargs)
        #: Time to wait between squeue calls.
        self.squeue_wait = squeue_wait
        #: Command to call squeue with.
        self.squeue_cmd = squeue_cmd
        #: Whether or not the thread should stop.
        self.stopped = threading.Event()
        #: Previous call to ``squeue``
        self.prev_call = 0.0
        #: Time to sleep between iterations in seconds.  Thread can only be
        #: terminated after this interval when waiting.
        self.sleep_time = sleep_time
        #: Maximal running time to accept for call to ``squeue``.
        self.squeue_timeout = squeue_timeout
        #: Maximal number of tries if call to ``squeue`` fails.
        self.max_tries = max_tries
        #: Dict mapping the job id to the job state string.
        self.states = {}
        self.cluster = cluster
        self.user = user
        #: Make at least one call to squeue, must not fail.
        logger.debug("initializing thread")
        self._call_squeue(allow_failure=False)
        self.prev_call = time.time()

    def _work(self):
        """Execute the thread's action"""
        while not self.stopped.is_set():
            now = time.time()
            if now - self.prev_call > self.squeue_wait:
                self._call_squeue()
                self.prev_call = now
            time.sleep(self.sleep_time)

    def get_state(self, jobid):
        """Return the job state for the given jobid."""
        jobid = str(jobid)
        if jobid not in self.states:
            # register job automatically
            self.register_job(jobid)
            self.states[jobid] = self._get_state_sacct(jobid)
        state = self.states.get(jobid, "__not_seen_yet__")
        logger.debug("job state of '%s': '%s'", jobid, state)

        retval = self._decide_return_value(jobid, state)
        logger.debug("returning state '%s' for job '%s'", state, jobid)
        return retval

    def register_job(self, jobid):
        """Register job with the given ID."""
        self.states.setdefault(jobid, None)

    def _decide_return_value(self, jobid, status, default="running"):
        if status.startswith("CANCELLED") or status in [
            "BOOT_FAIL",
            "OUT_OF_MEMORY",
            "DEADLINE",
            "FAILED",
            "NODE_FAIL",
            "TIMEOUT",
            "PREEMPTED",
        ]:
            logger.debug(f"Job '{jobid}' failed: '{status}'")
            return "failed"
        elif status in [
            "PENDING",
            "RUNNING",
            "REQUEUED",
            "COMPLETING",
            "SUSPENDED",  # Unclear whether SUSPENDED should be treated as running or failed
        ]:
            logger.debug(f"Job '{jobid}' running: '{status}'")
            return "running"
        elif status == "COMPLETED":
            logger.debug(f"Job '{jobid}' finished: '{status}'")
            return "success"
        else:
            logger.warning(f"Unkonwn job state of job '{jobid}': '{status}'")
            return default

    def _get_state_sacct(self, jobid):
        """Implement retrieving state via sacct for resuming jobs."""
        cluster = self.cluster
        cmd = ["sacct", "-P", "-b", "-j", jobid, "-n"]
        if cluster is not None:
            cmd.append("--cluster={cluster}".format(self.cluster))
        try_num = 0
        while try_num < self.max_tries:
            try_num += 1
            try:
                logger.debug("Calling %s (try %d)", cmd, try_num)
                output = subprocess.check_output(
                    cmd, timeout=self.squeue_timeout, universal_newlines=True
                )
                break
            except subprocess.TimeoutExpired as e:
                logger.debug(
                    "Call to %s timed out (try %d of %d)", cmd, try_num, self.max_tries
                )
            except subprocess.CalledProcessError as e:
                logger.debug(
                    "Call to %s failed (try %d of %d)", cmd, try_num, self.max_tries
                )
        if try_num >= self.max_tries:
            raise Exception("Problem with call to %s" % cmd)
        else:
            try:
                parsed = {
                    x.split("|")[0]: x.split("|")[1] for x in output.strip().split("\n")
                }
                logger.debug("Returning state of %s as %s", jobid, parsed[jobid])
                return parsed[jobid]
            except IndexError as e:
                logging.exception(
                    "Failed to parse sacct output for job %s: '%s'\n\tCommand: '%s'\n\tTry: %s\n\tError: %s",
                    jobid,
                    output,
                    cmd,
                    try_num,
                    e
                )
                return "UNKNOWN"

    def stop(self):
        """Flag thread to stop execution"""
        logger.debug("stopping thread")
        self.stopped.set()

    def _call_scontrol(self, jobid, allow_failure=True):
        """Run the call to ``scontrol`` to check whether a job gets requeued"""
        cluster = self.cluster

        cmd = ["scontrol", "-o", "show", "job", jobid]
        if cluster is not None:
            cmd.append("--cluster={cluster}".format(self.cluster))

        try_num = 0
        while try_num < self.max_tries:
            try_num += 1
            try:
                logger.debug("Calling %s (try %d)", cmd, try_num)
                output = subprocess.check_output(
                    cmd, timeout=self.squeue_timeout, universal_newlines=True
                )
                logger.debug("Output is:\n---\n%s\n---", output)
                break
            except subprocess.TimeoutExpired as e:
                if not allow_failure:
                    raise
                logger.debug(
                    "Call to %s timed out (try %d of %d)", cmd, try_num, self.max_tries
                )
            except subprocess.CalledProcessError as e:
                if not allow_failure:
                    raise
                logger.debug(
                    "Call to %s failed (try %d of %d)", cmd, try_num, self.max_tries
                )
        if try_num >= self.max_tries:
            logger.debug("Giving up for this round")
            return None
        else:
            logger.debug("parsing output")

            m = re.search(r"JobState=(\w+)", output.decode())
            res = m.group(1)

            if res == "PREEMPTED":
                m = re.search(r"Requeue=(\w+)", output.decode())
                requeueable = m.group(1)

                if requeueable == "1":
                    res = "REQUEUED"

            return res

    def _call_squeue(self, allow_failure=True):
        """Run the call to ``squeue``"""
        cluster = self.cluster
        try_num = 0
        cmd = [
            SQUEUE_CMD,
            "--format=%i,%T",
            "--state=all",
        ]
        if self.user is not None:
            cmd.append("--user={}".format(self.user))
        if cluster is not None:
            cmd.append("--cluster={cluster}".format(self.cluster))
        while try_num < self.max_tries:
            try_num += 1
            try:
                logger.debug("Calling %s (try %d)", cmd, try_num)
                output = subprocess.check_output(
                    cmd, timeout=self.squeue_timeout, universal_newlines=True
                )
                logger.debug("Output is:\n---\n%s\n---", output)
                break
            except subprocess.TimeoutExpired as e:
                if not allow_failure:
                    raise
                logger.debug(
                    "Call to %s timed out (try %d of %d)", cmd, try_num, self.max_tries
                )
            except subprocess.CalledProcessError as e:
                if not allow_failure:
                    raise
                logger.debug(
                    "Call to %s failed (try %d of %d)", cmd, try_num, self.max_tries
                )
        if try_num >= self.max_tries:
            logger.debug("Giving up for this round")
        else:
            logger.debug("parsing output")
            self._parse_output(output)

    def _parse_output(self, output):
        """Parse output of ``squeue`` call."""
        header = None
        for line in output.splitlines():
            line = line.strip()
            arr = line.split(",")
            if not header:
                if not line.startswith("JOBID"):
                    continue  # skip leader
                header = arr
            else:
                logger.debug("Updating state of %s to %s", arr[0], arr[1])
                self.states[arr[0]] = arr[1]

                # check if job was preempted
                if arr[1] == "PREEMPTED":
                    sctrl_res = self._call_scontrol(arr[0])
                    if sctrl_res is not None:
                        self.states[arr[0]] = sctrl_res


class JobStateHttpHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler class that responds to ```/job/status/${jobid}/`` GET requests"""

    def do_GET(self):
        """Only to ``/job/status/${job_id}/?``"""
        logger.debug("--- BEGIN GET")
        # Remove trailing slashes from path.
        path = self.path
        while path.endswith("/"):
            path = path[:-1]
        # Ensure that /job/status was requested
        if not self.path.startswith("/job/status/"):
            self.send_response(400)
            self.end_headers()
            return
        # Ensure authentication bearer is correct
        auth_required = "Bearer %s" % self.server.http_secret
        auth_header = self.headers.get("Authorization")
        logger.debug(
            "Authorization header is %s, required: %s"
            % (repr(auth_header), repr(auth_required))
        )
        if auth_header != auth_required:
            self.send_response(403)
            self.end_headers()
            return
        # Otherwise, query job ID status
        job_id = self.path[len("/job/status/") :]
        logger.debug("Querying for job ID %s" % repr(job_id))

        import time
        from datetime import timedelta

        start = time.time()
        status = self.server.poll_thread.get_state(job_id)
        elapsed = time.time() - start
        logger.debug(
            "Queried status for job %s in %s", job_id, timedelta(seconds=elapsed)
        )

        logger.debug("Status: %s" % status)
        if not status:
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            output = status + "\n"
            logger.debug("Sending '%s'" % repr(output))
            self.wfile.write(output.encode("utf-8"))
        logger.debug("--- END GET")

    def do_POST(self):
        """Handle POSTs (only to ``/job/register/${job_id}/?``)"""
        logger.debug("--- BEGIN POST")
        # Remove trailing slashes from path.
        path = self.path
        while path.endswith("/"):
            path = path[:-1]
        # Ensure that /job/register was requested
        if not self.path.startswith("/job/register/"):
            self.send_response(400)
            self.end_headers()
            return
        # Ensure authentication bearer is correct
        auth_required = "Bearer %s" % self.server.http_secret
        auth_header = self.headers.get("Authorization")
        logger.debug(
            "Authorization header is %s, required: %s",
            repr(auth_header),
            repr(auth_required),
        )
        # Otherwise, register job ID
        job_id = self.path[len("/job/status/") :]
        self.server.poll_thread.register_job(job_id)
        self.send_response(200)
        self.end_headers()
        logger.debug("--- END POST")

    def log_request(self, *args, **kwargs):
        if LOG_REQUESTS:
            super().log_request(*args, **kwargs)


class JobStateHttpServer(http.server.HTTPServer):
    """The HTTP server class"""

    allow_reuse_address = False

    def __init__(self, poll_thread):
        """Initialize thread and print the ``SNAKEMAKE_CLUSTER_SIDECAR_VARS`` to stdout, then flush."""
        super().__init__(("0.0.0.0", 0), JobStateHttpHandler)
        #: The ``PollSqueueThread`` with the state dictionary.
        self.poll_thread = poll_thread
        #: The secret to use.
        self.http_secret = str(uuid.uuid4())
        sidecar_vars = {
            "server_port": self.server_port,
            "server_secret": self.http_secret,
            "pid": os.getpid(),
        }
        logger.debug(json.dumps(sidecar_vars))

        if USE_CURL:
            command = """curl -s -X {method} -H {headers} {uri}"""

            method = "GET"

            # $0 will be replaced with the first command argument
            uri = "http://localhost:%d/job/status/" % (sidecar_vars["server_port"])

            headers = {"Authorization": "Bearer %s" % sidecar_vars["server_secret"]}
            headers = ['"{0}: {1}"'.format(k, v) for k, v in headers.items()]
            headers = " -H ".join(headers)

            command = command.format(method=method, headers=headers, uri=uri)

            sys.stdout.write(command + "\n")
        else:
            sys.stdout.write(json.dumps(sidecar_vars) + "\n")
        sys.stdout.flush()

    def log_message(self, *args, **kwargs):
        """Log messages are printed if ``DEBUG`` is ``True``."""
        if DEBUG:
            super().log_message(*args, **kwargs)


def main():
    # Start thread to poll ``squeue`` in a controlled fashion.
    poll_thread = PollSqueueThread(
        SQUEUE_WAIT,
        SQUEUE_CMD,
        cluster=CLUSTER,
        user=USER,
        name="poll-squeue",
    )
    poll_thread.start()

    # Initialize HTTP server that makes available the output of ``squeue --user [user]``
    # in a controlled fashion.
    http_server = JobStateHttpServer(poll_thread)
    http_thread = threading.Thread(name="http-server", target=http_server.serve_forever)
    http_thread.start()

    # Allow for graceful shutdown of poll thread and HTTP server.
    def signal_handler(signum, frame):
        """Handler for Unix signals. Shuts down http_server and poll_thread."""
        logger.info("Shutting down squeue poll thread and HTTP server...")
        # from remote_pdb import set_trace
        # set_trace()
        poll_thread.stop()
        http_server.shutdown()
        logger.info("... HTTP server and poll thread shutdown complete.")
        for thread in threading.enumerate():
            logger.info("ACTIVE %s", thread.name)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Actually run the server.
    poll_thread.join()
    logger.debug("poll_thread done")
    http_thread.join()
    logger.debug("http_thread done")


if __name__ == "__main__":
    sys.exit(int(main() or 0))
