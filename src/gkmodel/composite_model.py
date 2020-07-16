"""
    composite_model
    ~~~~~~~~~~~~~~~
"""
from typing import List, Union, Dict, Tuple
import pandas as pd
import numpy as np

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

    def _get_cov_names(self, cov_models):
        cov_names = []
        for cov_model in cov_models:
            cov_names.extend(cov_model.covs)
        return cov_names

    def _get_stage_data(self, data: MRData):
        pred = self.fitted_models[-1].predict(data)
        resi = data.obs - pred
        df = data.to_df()
        df['resi'] = resi
        data_next = MRData()
        cov_names = []
        for i in range(len(self.sub_models)):
            cov_names.extend(self._get_cov_names(self.sub_models[i][1]))
        data_next.load_df(df, col_covs=cov_names, col_obs='resi', col_obs_se='obs_se', col_study_id='study_id')
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
        model = self.fitted_models[i]
        if model.__class__.__name__ == 'OverallModel':
            names = []
            for cov_model in model.cov_models:
                names.extend([cov_model.name + '_' + str(i) for i in range(cov_model.num_x_vars)])
            assert len(names) == len(model.soln)
            df = pd.DataFrame(list(zip(names, model.soln)), columns=['name', 'value'])
        else:
            names = self._get_cov_names(model.cov_models)
            df = pd.DataFrame.from_dict(model.soln, orient='index', columns=names).reset_index().rename(columns={'index': 'study_id'})
        
        if path is not None:
            df.to_csv(path)
        return df
