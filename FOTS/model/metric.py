import numpy as np
from ..utils.eval_tools.icdar2015 import eval as icdar_eval


def fots_metric(pred, gt):
    config = icdar_eval.default_evaluation_params()
    output = icdar_eval.eval(pred, gt, config)
    return output['method']['precision'], output['method']['recall'], output['method']['hmean']