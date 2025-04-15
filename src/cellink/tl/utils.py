import logging
import sys

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def _get_vep_start_row(filename):
    with open(filename) as file:
        for line_number, line in enumerate(file, start=0):
            if line.startswith("#") and not line.startswith("##"):
                return line_number
