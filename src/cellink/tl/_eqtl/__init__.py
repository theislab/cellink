from ._data import EQTLData
from ._pipeline import EQTLPipeline
from ._utils import bonferroni_adjustment, q_value, quantile_transform

pbdata_transforms_dict = {
    "none": None,
    "quantile_transform": quantile_transform,
}

pv_transforms_dict = {"none": None, "bonferroni_adjustment": bonferroni_adjustment, "q_value": q_value}
