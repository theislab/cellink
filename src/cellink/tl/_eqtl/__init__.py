from ._data import EQTLData
from ._pipeline import EQTLPipeline
from ._utils import quantile_transform, bonferroni_adjustment, q_value

pbdata_transforms_dict = {
    "none": None,
    "quantile_transform": quantile_transform,
}

pv_transforms_dict = {
    "none": None,
    "bonferroni_adjustment": bonferroni_adjustment,
    "q_value": q_value
}
