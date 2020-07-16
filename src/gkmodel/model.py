"""
    model
    ~~~~~
"""
from pathlib import Path
from typing import List, Union, Dict
import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass, field, asdict

from mrtool import MRData, LinearCovModel


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


class OverallModel:
    """Overall model in charge of fit all location together without
    random effects.
    """

    def __init__(self,
                 data: MRData,
                 cov_models: List[LinearCovModel]):
        """Constructor of OverallModel

        Args:
            data (MRData): Data object from MRTool
            cov_models (List[LinearCovModel]):
                List of linear covariate model from MRTool.
        """
        self.data = data
        self.cov_models = cov_models
        for cov_model in self.cov_models:
            cov_model.attach_data(self.data)

        self.mat = self.create_design_mat()
        self.soln = None

    def create_design_mat(self, data: MRData = None) -> np.ndarray:
        """Create design matrix

        Args:
            data (MRData, optional):
                Create design matrix from the given data object. If ``None`` use
                the attribute ``self.data``. Defaults to None.

        Returns:
            np.ndarray: Design matrix.
        """
        data = self.data if data is None else data
        return np.hstack([cov_model.create_design_mat(data)[0]
                          for cov_model in self.cov_models])

    def fit_model(self):
        """Fit the model
        """
        self.soln = solve_ls(self.mat, self.data.obs, self.data.obs_se)

    def predict(self, data: MRData = None) -> np.ndarray:
        """Predict from fitting result.

        Args:
            data (MRData, optional):
                Given data object to predict, if ``None`` use the attribute
                ``self.data`` Defaults to None.

        Returns:
            np.ndarray: Prediction.
        """
        data = self.data if data is None else data
        mat = self.create_design_mat(data)
        return mat.dot(self.soln)

    def write_soln(self, path: str = None):
        names = []
        for cov_model in self.cov_models:
            names.extend([cov_model.name + '_' + str(i)
                          for i in range(cov_model.num_x_vars)])
        assert len(names) == len(self.soln)
        df = pd.DataFrame(list(zip(names, self.soln)),
                          columns=['name', 'value'])
        if path is not None:
            df.to_csv(path)
        return df


class StudyModel:
    """Study specific Model.
    """

    def __init__(self, data: MRData, cov_models: List[LinearCovModel]):
        """Constructor of StudyModel

        Args:
            data (MRData): MRTool data object.
            cov_models (List[LinearCovModel]):
                List of linear covariate model from MRTool.
        """
        self.data = data
        self.cov_names = self._get_cov_names(cov_models)
        self.mat = self.create_design_mat()
        self.soln = None

    def _get_cov_names(self, cov_models):
        cov_names = []
        for cov_model in cov_models:
            cov_names.extend(cov_model.covs)
        return cov_names

    def create_design_mat(self, data: MRData = None) -> np.ndarray:
        """Create design matrix.

        Args:
            data (MRData, optional):
                Create design matrix from the given data object. If ``None`` use
                the attribute ``self.data``. Defaults to None. Defaults to None.

        Returns:
            np.ndarray: Design matrix.
        """
        data = self.data if data is None else data
        mat = data.get_covs(self.cov_names)

        return mat

    def fit_model(self):
        """Fit the model.
        """
        self.soln = {}
        for study_id in self.data.studies:
            index = self.data.study_id == study_id
            mat = self.mat[index, :]
            obs = self.data.obs[index]
            obs_se = self.data.obs_se[index]
            self.soln[study_id] = solve_ls(mat, obs, obs_se)

    def predict(self, data: MRData = None, slope_quantile: Dict[str, float] = None) -> np.ndarray:
        """Predict from fitting result.

        Args:
            data (MRData, optional):
                Given data object to predict, if ``None`` use the attribute
                ``self.data`` Defaults to None.

        Returns:
            np.ndarray: Prediction.
        """
        data = self.data if data is None else data
        mat = self.mat if data is None else self.create_design_mat(data)

        mean_soln = np.mean(list(self.soln.values()), axis=0)

        soln = np.array([
            self.soln[study_id]
            if study_id in self.data.study_id else mean_soln
            for study_id in data.study_id
        ])

        if slope_quantile is not None:
            for name, quantile in slope_quantile.items():
                if name not in self.cov_names:
                    raise ValueError(
                        f'{name} is not in covariates for study model, which are {self.cov_names}.')
                i = self.cov_names.index(name)
                v = np.quantile(soln[:, i], quantile)
                if quantile >= 0.5:
                    soln[:, i] = np.maximum(soln[:, i], v)
                else:
                    soln[:, i] = np.minimum(soln[:, i], v)

        return np.sum(mat*soln, axis=1)

    def write_soln(self, path: str = None):
        df = pd.DataFrame.from_dict(
            self.soln,
            orient='index',
            columns=self.cov_names
        ).reset_index().rename(columns={'index': 'study_id'})
        if path is not None:
            df.to_csv(path)
        return df


def solve_ls(mat: np.ndarray,
             obs: np.ndarray, obs_se: np.ndarray) -> np.ndarray:
    """Solve least square problem

    Args:
        mat (np.ndarray): Data matrix
        obs (np.ndarray): Observations
        obs_se (np.ndarray): Observation standard error.

    Returns:
        np.ndarray: Solution.
    """
    v = obs_se**2
    return np.linalg.solve((mat.T/v).dot(mat),
                           (mat.T/v).dot(obs))


def result_to_df(
    model: Union[OverallModel, StudyModel],
    data: MRData,
    prediction: str = 'prediction',
    residual: str = 'residual'
) -> pd.DataFrame:
    """Create result data frame.

    Args:
        model (Union[OverallModel, StudyModel]): Model instance.
        prediction (str, optional):
            Column name of the prediction. Defaults to 'prediction'.
        residual (str, optional):
            Column name of the residual. Defaults to 'residual'.

    Returns:
        pd.DataFrame: Result data frame.
    """
    data._sort_by_data_id()
    pred = model.predict(data)
    resi = data.obs - pred
    df = data.to_df()
    df[prediction] = pred
    df[residual] = resi

    return df
