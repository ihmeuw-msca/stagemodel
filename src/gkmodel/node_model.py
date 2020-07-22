"""
    model
    ~~~~~
"""
from typing import List, Union, Dict, Tuple, Any
import numpy as np
import pandas as pd
from copy import deepcopy

from mrtool import MRData, LinearCovModel

from .utils import solve_ls, result_to_df


class NodeModel:
    """Node model that carries independent task.
    """

    def __init__(self,
                 data: MRData = None,
                 cov_models: List[LinearCovModel] = None):
        """Constructor of the NodeModel.

        Args:
            data (MRData):
                Data object from MRTool. If ``None``, no data is attached.
                Default to ``None``.
            cov_models (List[LinearCovModel]):
                List of linear covariate model from MRTool. If ``None``,
                intercept model will be added. Default to ``None``.
        """
        self.data = None
        self.cov_models = [LinearCovModel('intercept')] if cov_models is None else cov_models
        self.cov_names = self.get_cov_names()
        self.mat = None
        self.soln = None

        self.attach_data(data)

    def attach_data(self, data: Union[MRData, None]):
        """Attach data into the model object.

        Args:
            data (Union[MRData, None]):
                Data object if ``None``, do nothing. Default to ``None``.
        """
        if data is not None:
            self.data = data
            for cov_model in self.cov_models:
                cov_model.attach_data(self.data)
            self.mat = self.create_design_mat()

    def _assert_has_data(self):
        """Assert attached data.

        Raises:
            ValueError: If attribute ``data`` is ``None``, return value error.
        """
        if self.data is None:
            raise ValueError("Must attach data!")

    def _assert_has_soln(self):
        """Assert has solution.

        Raises:
            ValueError: If attribute ``soln`` is ``None``, return value error.
        """
        if self.soln is None:
            raise ValueError("Must fit model!")

    def get_cov_names(self,
                      cov_models: List[LinearCovModel] = None) -> List[str]:
        """Get covariates names.

        Args:
            cov_models (List[LinearCovModel], optional):
                List of covariate models. If ``None`` ues the attribute
                ``cov_models``. Defaults to None.

        Returns:
            List[str]: List of covariates names.
        """
        cov_models = self.cov_models if cov_models is None else cov_models
        cov_names = []
        for cov_model in self.cov_models:
            cov_names.extend(cov_model.covs)
        return cov_names

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
        """Fit the model.
        """
        raise NotImplementedError()

    def predict(self, data: MRData = None, **kwargs) -> np.ndarray:
        """Predict from fitting result.

        Args:
            data (MRData, optional):
                Given data object to predict, if ``None`` use the attribute
                ``self.data`` Defaults to None.
            kwargs (Dict): Other keyword arguments.

        Returns:
            np.ndarray: Prediction.
        """
        raise NotImplementedError()

    def soln_to_df(self, path: Union[str, None] = None) -> pd.DataFrame:
        """Write the soln to the path.

        Args:
            path (Union[str, None], optional):
                Address that save the result, include the file name.
                If ``None`` do not save the result, only return the result data
                frame. Defaults to None.

        Returns:
            pd.DataFrame: Data frame that contains the result.
        """
        raise NotImplementedError()

    def result_to_df(self,
                     path: str = None,
                     prediction: str = 'prediction',
                     residual: str = 'residual') -> pd.DataFrame:
        """Create result data frame.

        Args:
            path (Union[str, None], optional):
                Address that save the result, include the file name.
                If ``None`` do not save the result, only return the result data
                frame. Defaults to None.
            prediction (str, optional):
                Column name of the prediction. Defaults to 'prediction'.
            residual (str, optional):
                Column name of the residual. Defaults to 'residual'.

        Returns:
            pd.DataFrame: Result data frame.
        """
        self._assert_has_data()
        self._assert_has_soln()
        return result_to_df(self, self.data,
                            path=path, prediction=prediction, residual=residual)


class OverallModel(NodeModel):
    """Overall model in charge of fit all location together without
    random effects.
    """

    def fit_model(self):
        """Fit the model
        """
        self._assert_has_data()
        self.soln = solve_ls(self.mat, self.data.obs, self.data.obs_se)

    def predict(self, data: MRData = None, **kwargs) -> np.ndarray:
        """Predict from fitting result.
        """
        self._assert_has_soln()
        data = self.data if data is None else data
        mat = self.create_design_mat(data)
        return mat.dot(self.soln)

    def soln_to_df(self, path: str = None) -> pd.DataFrame:
        """Write solution.
        """
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


class StudyModel(NodeModel):
    """Study specific Model.
    """

    def fit_model(self):
        """Fit the model.
        """
        self._assert_has_data()
        self.soln = {}
        for study_id in self.data.studies:
            index = self.data.study_id == study_id
            mat = self.mat[index, :]
            obs = self.data.obs[index]
            obs_se = self.data.obs_se[index]
            self.soln[study_id] = solve_ls(mat, obs, obs_se)

    def predict(
        self,
        data: MRData = None,
        slope_quantile: Dict[str, float] = None,
        ref_cov: Tuple[str, Any] = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict from fitting result.

        Args:
            slope_quantile (Dict[str, float]):
                Dictionary with key as the covariate name and value the
                quantile. If ``None`` will predict for specific group, else
                use the quantile or more extreme slope. Default to ``None``.
        """
        self._assert_has_soln()
        data = self.data if data is None else data
        data._sort_by_data_id()
        mat = self.mat if data is None else self.create_design_mat(data)

        mean_soln = np.mean(list(self.soln.values()), axis=0)

        soln = np.array([
            self.soln[study_id]
            if study_id in self.data.study_id else mean_soln
            for study_id in data.study_id
        ])

        adjust_values = np.zeros(self.data.num_points)

        if slope_quantile is not None:
            covs_index = []
            quantiles = []
            for name, quantile in slope_quantile.items():
                if name in self.cov_names:
                    covs_index.append(self.cov_names.index(name))
                    quantiles.append(quantile)

            if len(covs_index) > 0:             
                if ref_cov is not None:
                    ref_mat = deepcopy(mat)
                    for study in self.data.studies:
                        study_index = self.data.study_id == study
                        ref_index = study_index & (self.data.covs[ref_cov[0]] == ref_cov[1])
                        if sum(ref_index) != 1:
                            raise RuntimeError('One and only one ref value per group allowed.')
                        ref_mat[study_index, covs_index] = ref_mat[ref_index, covs_index]
                    
                    ref_before_values = np.sum(ref_mat * soln, axis=1)

                for i, quantile in zip(covs_index, quantiles):
                    v = np.quantile(soln[:, i], quantile)
                    if quantile >= 0.5:
                        soln[:, i] = np.maximum(soln[:, i], v)
                    else:
                        soln[:, i] = np.minimum(soln[:, i], v)

                if ref_cov is not None:
                    ref_after_values = np.sum(ref_mat * soln, axis=1)
                    adjust_values = ref_after_values - ref_before_values
        
        return np.sum(mat*soln, axis=1) - adjust_values

    def soln_to_df(self, path: str = None) -> pd.DataFrame:
        """Write solution.
        """
        df = pd.DataFrame.from_dict(
            self.soln,
            orient='index',
            columns=self.cov_names
        ).reset_index().rename(columns={'index': 'study_id'})
        if path is not None:
            df.to_csv(path)
        return df
