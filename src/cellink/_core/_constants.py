from typing import NamedTuple

# scVI Manager Store Constants
# ----------------------------
# Keys for UUIDs used for referencing model class manager stores.

_SCVI_UUID_KEY = "_scvi_uuid"
_MANAGER_UUID_KEY = "_scvi_manager_uuid"

# scVI Registry Constants
# -----------------------
# Keys used in the scVI registry.

_SCVI_VERSION_KEY = "scvi_version"
_MODEL_NAME_KEY = "model_name"
_SETUP_METHOD_NAME = "setup_method_name"
_SETUP_ARGS_KEY = "setup_args"
_FIELD_REGISTRIES_KEY = "field_registries"
_DATA_REGISTRY_KEY = "data_registry"
_STATE_REGISTRY_KEY = "state_registry"
_SUMMARY_STATS_KEY = "summary_stats"

# scVI Data Registry Constants
# ----------------------------
# Keys used in the data registry.

_DR_MOD_KEY = "mod_key"
_DR_ATTR_NAME = "attr_name"
_DR_ATTR_KEY = "attr_key"

# AnnData Minification Constants
# ------------------------
# Constants used in handling adata minification.

class _REGISTRY_KEYS_NT(NamedTuple):
    DONOR_X_KEY: str = "donor_X"
    CELL_X_KEY: str = "cell_X"
    DONOR_LABELS_KEY: str = "donor_labels"
    DONOR_BATCH_KEY: str = "donor_batch"
    DONOR_CAT_COVS_KEY: str = "donor_extra_categorical_covs"
    DONOR_CONT_COVS_KEY: str = "donor_extra_continuous_covs"
    DONOR_INDICES_KEY: str = "donor_ind_x"
    DONOR_PATIENT_KEY: str = "donor_patient"
    CELL_LABELS_KEY: str = "cell_labels"
    CELL_BATCH_KEY: str = "cell_batch"
    CELL_CAT_COVS_KEY: str = "cell_extra_categorical_covs"
    CELL_CONT_COVS_KEY: str = "cell_extra_continuous_covs"
    CELL_INDICES_KEY: str = "cell_ind_x"
    CELL_PATIENT_KEY: str = "cell_patient"

REGISTRY_KEYS = _REGISTRY_KEYS_NT()
