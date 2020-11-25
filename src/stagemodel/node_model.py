"""
    model
    ~~~~~
"""
from typing import List, Union, Dict, Tuple, Any
import numpy as np
import pandas as pd
import xarray as xr
from warnings import warn

from mrtool import MRData, LinearCovModel, MRBRT

from .utils import solve_ls, solve_ls_b, result_to_df


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

    def predict_from_xarray(self, covs: List[xr.DataArray], **kwargs) -> xr.DataArray:
        """Predict from xarray

        Args:
            data (xr.Dataset): xarray dataset that contains all the covariates.

        Returns:
            xr.Dataset: prediction result
        """
        pred = self.predict_from_intercept(**kwargs)
        for cov in covs:
            pred = pred + self.predict_from_xarray_cov(cov, **kwargs)
        return pred

    def predict_from_xarray_cov(self, cov: xr.DataArray, **kwargs) -> xr.DataArray:
        covs = {cov.name: cov.data.ravel()}
        covs.update({
            cov_name: np.zeros(cov.size)
            for cov_name in self.cov_names if cov_name != cov.name
        })
        pred = self.predict(MRData(covs=covs), **kwargs).reshape(cov.shape)
        return xr.DataArray(pred, name=f"pred_{cov.name}", coords=cov.coords, dims=cov.dims)

    def predict_from_intercept(self, **kwargs) -> float:
        covs = {"intercept": np.array([1.0])}
        covs.update({
            cov_name: np.zeros(1)
            for cov_name in self.cov_names if cov_name != "intercept"
        })
        return self.predict(MRData(covs=covs), **kwargs)[0]

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

    def fit_model(self, **fit_options):
        """Fit the model
        """
        self._assert_has_data()
        beta_init = solve_ls(self.mat, self.data.obs, self.data.obs_se)
        model = MRBRT(self.data, self.cov_models)
        gamma_init = np.zeros(model.num_z_vars)

        default_fit_options = dict(
            x0=np.hstack((beta_init, gamma_init)),
            inner_max_iter=500,
            inner_print_level=5,
        )
        fit_options = {**default_fit_options, **fit_options}
        model.fit_model(**fit_options)
        self.soln = model.beta_soln

    def predict(self, data: MRData = None, **kwargs) -> np.ndarray:
        """Predict from fitting result.
        """
        self._assert_has_soln()
        data = self.data if data is None else data
        data._sort_by_data_id()
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
        bounds = np.vstack([cov_model.prior_beta_uniform.T
                            for cov_model in self.cov_models])
        use_bounds = (not np.isneginf(bounds[:, 0]).all()) or \
            (not np.isposinf(bounds[:, 1]).all())
        gprior = np.hstack([cov_model.prior_beta_gaussian
                            for cov_model in self.cov_models])
        if np.isinf(gprior[1]).all():
            gprior = None
        for study_id in self.data.studies:
            index = self.data.study_id == study_id
            mat = self.mat[index, :]
            obs = self.data.obs[index]
            obs_se = self.data.obs_se[index]
            if use_bounds:
                self.soln[study_id] = solve_ls_b(mat, obs, obs_se, bounds,
                                                 gprior=gprior)
            else:
                self.soln[study_id] = solve_ls(mat, obs, obs_se,
                                               gprior=gprior)

        self.soln['mean'] = np.array(list(self.soln.values())).mean(axis=0)

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

        if slope_quantile is not None:
            _, soln = self.get_soln_quantile(slope_quantile, mask_soln=True)
        else:
            soln = self.soln

        coefs = np.vstack([
            soln[study_id]
            if study_id in self.data.studies else soln['mean']
            for study_id in data.study_id
        ])

        intercept_shift = {study_id: 0.0 for study_id in data.studies}
        if ref_cov is not None:
            for study_id in data.studies:
                sub_mat = mat[(data.study_id == study_id) &
                              (data.covs[ref_cov[0]] == ref_cov[1])]
                if sub_mat.shape[0] != 1:
                    warn(f'Multiple ref value for study {study_id} found. Using mean instead.')
                study_name = study_id if study_id in self.data.studies else 'mean'
                intercept_shift[study_id] = np.mean(sub_mat.dot(self.soln[study_name] - soln[study_name]))

        shifts = np.array([intercept_shift[study_id] for study_id in data.study_id])

        return np.sum(mat*coefs, axis=1) + shifts

    def get_soln_quantile(self,
                          slope_quantile: Dict[str, float],
                          mask_soln: bool = False) -> Union[Dict[str, float],
                                                            Tuple[Dict[str, float], Dict[Any, np.ndarray]]]:
        """Get solution quantile

        Args:
            slope_quantile (Dict[str, float]): Solution quantile.
            mask_soln (bool, optional):
                If ``True``, return masked solution. Default to ``False``.

        Returns:
            Union[Dict[str, float], Tuple[Dict[str, float], Dict[Any, np.ndarray]]]:
                Quantile of the solution or with the masked solution.
        """
        coefs = np.array([self.soln[study_id]
                          for study_id in self.data.studies])
        quantile_value = {}
        for cov_name, q in slope_quantile.items():
            if cov_name not in self.cov_names:
                warn(f"{cov_name} not in the model, "
                     f"ignore it in slope_quantile.")
            else:
                quantile_value[cov_name] = np.quantile(
                    coefs[:, self.cov_names.index(cov_name)], q
                )
        if mask_soln:
            masked_coefs = np.vstack([coefs, self.soln['mean']])
            for cov_name, v in quantile_value.items():
                index = self.cov_names.index(cov_name)
                if slope_quantile[cov_name] >= 0.5:
                    masked_coefs[:, index] = np.maximum(masked_coefs[:, index], v)
                else:
                    masked_coefs[:, index] = np.minimum(masked_coefs[:, index], v)
            masked_soln = {
                study_id: masked_coefs[i]
                for i, study_id in enumerate(self.data.studies)
            }
            masked_soln['mean'] = masked_coefs[-1]
            return quantile_value, masked_soln
        else:
            return quantile_value

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
