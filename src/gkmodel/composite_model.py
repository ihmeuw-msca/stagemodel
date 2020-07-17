"""
    composite_model
    ~~~~~~~~~~~~~~~
"""
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from copy import deepcopy

from mrtool import MRData, LinearCovModel
from .model import OverallModel, StudyModel


class StagewiseModel:

    def __init__(
        self,
        data: MRData,
        sub_models: List[Tuple[str, List[LinearCovModel]]],
    ):
        self.sub_models = sub_models
        self.num_sub_models = len(sub_models)
        self.datas = [data]
        self.fitted_models = []

    def _get_stage_data(self, data: MRData):
        pred = self.fitted_models[-1].predict(self.datas[-1])
        resi = self.datas[-1].obs - pred
        data_next = deepcopy(self.datas[-1])
        data_next.obs = resi

        self.datas.append(data_next)

    def fit_model(self):
        while len(self.sub_models) > 0:
            mtype, covmodels = self.sub_models.pop(0)
            if mtype == 'overall':
                self.fitted_models.append(OverallModel(self.datas[-1], covmodels))
            elif mtype == 'study':
                self.fitted_models.append(StudyModel(self.datas[-1], covmodels))
            else:
                raise ValueError(f'model type {mtype} is invalid.')
            self.fitted_models[-1].fit_model()
            if len(self.sub_models) > 0:
                self._get_stage_data(self.datas[-1])

    def predict(self, data: MRData = None, slope_quantile: Dict[str, float] = None):
        if data is None:
            data = self.datas[0]
        data._sort_by_data_id()
        pred = np.zeros(data.num_obs)
        for model in self.fitted_models:
            if model.__class__.__name__ == 'OverallModel':
                pred += model.predict(data)
            else:
                pred += model.predict(data, slope_quantile=slope_quantile)
        return pred

    def write_soln(self, i, path: str = None):
        return self.fitted_models[i].write_soln()


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

    def predict(self, data: MRData = None,
                slope_quantile: Dict[str, float] = None):
        if data is None:
            data = self.data1
        data._sort_by_data_id()
        pred1 = self.model1.predict(data)
        return self.model2.predict(data, slope_quantile=slope_quantile) + pred1

    def write_soln(self, path: str = "./"):
        """Write the solutions.

        Args:
            path (str, optional): [description]. Defaults to "./".
        """
        base_path = Path(path)
        self.model1.write_soln(base_path / "result1.csv")
        self.model2.write_soln(base_path / "result2.csv")
