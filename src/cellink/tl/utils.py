import logging
import sys

import pandas as pd

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


def _explode_columns(annos, col_to_explode):
    logger.info(f"Exploding column {col_to_explode}")
    annos_explode = annos.copy()
    annos_explode[col_to_explode] = annos_explode[col_to_explode].str.split(",")
    annos_explode = annos_explode.explode(col_to_explode)
    logger.info(f"Number of original rows: {len(annos)}.\n Number of exploded rows: {len(annos_explode)}")
    return annos_explode


def _add_dummy_cols(annos, col_to_dummy):
    logger.info(f"Making dummies from column {col_to_dummy}")
    annos_with_dummy = annos.copy()
    dum = pd.get_dummies(annos_with_dummy[col_to_dummy], prefix=col_to_dummy, prefix_sep="_", dtype=int)
    annos_with_dummy = pd.concat([annos_with_dummy.drop(columns=col_to_dummy), dum], axis=1)
    return annos_with_dummy
