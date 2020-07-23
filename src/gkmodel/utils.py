"""
    utils
    ~~~~~
"""
from typing import Dict, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult


def solve_ls(mat: np.ndarray,
             obs: np.ndarray, obs_se: np.ndarray) -> np.ndarray:
    """Solve least square problem

    Args:
        mat(np.ndarray): Data matrix
        obs(np.ndarray): Observations
        obs_se(np.ndarray): Observation standard error.

    Returns:
        np.ndarray: Solution.
    """
    v = obs_se**2
    return np.linalg.solve((mat.T/v).dot(mat),
                           (mat.T/v).dot(obs))


def solve_ls_b(mat: np.ndarray,
               obs: np.ndarray, obs_se: np.ndarray,
               bounds: np.ndarray,
               options: Dict = None,
               return_info: bool = False) -> Union[np.ndarray, OptimizeResult]:
    """Solve least square with bounds problem

    Args:
        mat(np.ndarray): Data matrix
        obs(np.ndarray): Observations
        obs_se(np.ndarray): Observation standard error.
        bounds(np.ndarray): Bounds for the variable.
        options(Dict, optional): Options for scipy solver. Default to None.
        return_info(bool, optional):
            If ``True``, return the convergence information.
            Default to ``False``.

    Returns:
        np.ndarray: Solution.
    """
    x_init = solve_ls(mat, obs, obs_se)
    v = obs_se**2

    def objective(x):
        r = obs - mat.dot(x)
        return 0.5*np.sum(r**2/v)

    def gradient(x):
        r = obs - mat.dot(x)
        return (mat.T/v).dot(r)

    opt_result = minimize(objective,
                          x0=x_init,
                          jac=gradient,
                          method='L-BFGS-B',
                          bounds=bounds,
                          options=options)

    result = opt_result if return_info else opt_result.x

    return result


def result_to_df(model, data,
                 path: str = None,
                 prediction: str = 'prediction',
                 residual: str = 'residual') -> pd.DataFrame:
    """Create result data frame.

    Args:
        model (Union[NodeModel, StagewiseModel]): Model instance.
        data (MRData): Data object try to predict.s
        prediction (str, optional):
            Column name of the prediction. Defaults to 'prediction'.
        residual (str, optional):
            Column name of the residual. Defaults to 'residual'.
        path (Union[str, None], optional):
                Address that save the result, include the file name.
                If ``None`` do not save the result, only return the result data
                frame. Defaults to None.

    Returns:
        pd.DataFrame: Result data frame.
    """
    data._sort_by_data_id()
    pred = model.predict(data)
    resi = data.obs - pred
    df = data.to_df()
    df[prediction] = pred
    df[residual] = resi

    if path is not None:
        df.to_csv(path)

    return df
