"""
    composite_model
    ~~~~~~~~~~~~~~~
"""
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union
import numpy as np
import xarray as xr
from copy import deepcopy

from mrtool import MRData, LinearCovModel
from .node_model import NodeModel, OverallModel, StudyModel
from .utils import result_to_df


class StagewiseModel:

    def __init__(self,
                 data: MRData,
                 node_models: List[NodeModel]):
        self.node_models = node_models
        self.num_models = len(node_models)
        self.data_list = [data]

    def _get_next_data(self, model: NodeModel):
        pred = model.predict(self.data_list[-1])
        resi = self.data_list[-1].obs - pred
        data = deepcopy(self.data_list[-1])
        data.obs = resi

        self.data_list.append(data)

    def fit_model(self):
        for i, model in enumerate(self.node_models):
            model.attach_data(self.data_list[-1])
            model.fit_model()
            if i + 1 < self.num_models:
                self._get_next_data(model)

    def predict(
        self,
        data: Union[MRData, List[xr.DataArray]] = None,
        slope_quantile: Dict[str, float] = None,
        ref_cov: Tuple[str, Any] = None,
    ) -> Union[np.ndarray, xr.DataArray]:
        if data is None:
            data = self.data_list[0]
        data._sort_by_data_id()
        pred = np.zeros(data.num_obs)
        for model in self.node_models:
            pred += model.predict(data, slope_quantile=slope_quantile, ref_cov=ref_cov)
        return pred

    def soln_to_df(self, i: int, path: str = None):
        return self.node_models[i].soln_to_df(path)

    def result_to_df(self, i: int = None, path: str = None,
                     prediction: str = 'prediction',
                     residual: str = 'residual'):
        if i is not None:
            df = self.node_models[i].result_to_df(path, prediction, residual)
        else:
            df = result_to_df(self, self.data_list[0],
                              path, prediction, residual)
        return df


class TwoStageModel:

    def __init__(
        self,
        data: MRData,
        cov_models_stage1: List[LinearCovModel],
        cov_models_stage2: List[LinearCovModel],
    ):
        self.cov_models1 = cov_models_stage1
        self.cov_models2 = cov_models_stage2

        self.model1 = None
        self.model2 = None

        self.data1 = data
        self.data2 = None

    def _get_stage2_data(self, data: MRData):
        pred = self.model1.predict(data)
        resi = data.obs - pred
        data2 = deepcopy(self.data1)
        data2.obs = resi
        return data2

    def fit_model(self):
        # -------- stage 1: calling overall model -----------
        self.model1 = OverallModel(self.data1, self.cov_models1)
        self.model1.fit_model()

        # ---------- stage 2: calling study model ----------
        self.data2 = self._get_stage2_data(self.data1)
        self.model2 = StudyModel(self.data2, self.cov_models2)
        self.model2.fit_model()

    def predict(
        self,
        data: MRData = None,
        slope_quantile: Dict[str, float] = None,
        ref_cov: Tuple[str, Any] = None,
    ):
        if data is None:
            data = self.data1
        data._sort_by_data_id()
        pred1 = self.model1.predict(data)
        return self.model2.predict(data, slope_quantile=slope_quantile, ref_cov=ref_cov) + pred1


class ReverseTwoStageModel:

    def __init__(
        self,
        data: MRData,
        cov_models_stage1: List[LinearCovModel],
        cov_models_stage2: List[LinearCovModel],
    ):
        self.cov_models1 = cov_models_stage1
        self.cov_models2 = cov_models_stage2

        self.model1 = None
        self.model2 = None

        self.data1 = data
        self.data2 = None

    def _get_stage2_data(self, data: MRData):
        pred = self.model1.predict(data)
        resi = data.obs - pred
        data2 = deepcopy(self.data1)
        data2.obs = resi
        return data2

    def fit_model(self):
        # -------- stage 1: calling study model -----------
        self.model1 = StudyModel(self.data1, self.cov_models1)
        self.model1.fit_model()

        # ---------- stage 2: calling overall model ----------
        self.data2 = self._get_stage2_data(self.data1)
        self.model2 = OverallModel(self.data2, self.cov_models2)
        self.model2.fit_model()

    def predict(
        self,
        data: MRData = None,
        slope_quantile: Dict[str, float] = None,
        ref_cov: Tuple[str, Any] = None,
    ):
        if data is None:
            data = self.data1
        data._sort_by_data_id()
        pred1 = self.model1.predict(data, slope_quantile=slope_quantile, ref_cov=ref_cov)
        return self.model2.predict(data) + pred1
