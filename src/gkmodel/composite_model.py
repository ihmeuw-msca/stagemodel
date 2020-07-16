"""
    composite_model
    ~~~~~~~~~~~~~~~
"""
from typing import List, Union, Dict, Tuple
from mrtool import MRData, LinearCovModel
from .model import OverallModel, StudyModel


class StagewiseModel:

    def __init__(
        self,
        data: MRData,
        sub_models: List[Tuple[str, List[LinearCovModel]]],
    ):
        self.num_sub_models = len(sub_models)
        self.data = data

    def _get_cov_names(self, cov_models):
        cov_names = []
        for cov_model in cov_models:
            cov_names.extend(cov_model.covs)
        return cov_names

    def _get_stage_data(self, data: MRData):
        pass

    def fit_model(self):
        pass

    def predict(self, data: MRData = None, slope_quantile: Dict[str, float] = None):
        pass

    def write_soln(self, path: str = None):
        pass
