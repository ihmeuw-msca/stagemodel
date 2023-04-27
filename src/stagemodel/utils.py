"""
    utils
    ~~~~~
"""
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize


def solve_ls(
    mat: np.ndarray, obs: np.ndarray, obs_se: np.ndarray, gprior: np.ndarray = None
) -> np.ndarray:
    """Solve least square problem

    Args:
        mat(np.ndarray): Data matrix
        obs(np.ndarray): Observations
        obs_se(np.ndarray): Observation standard error.
        gprior(np.ndarray): Gaussian prior. Default to ``None``.

    Returns:
        np.ndarray: Solution.
    """
    v = obs_se**2
    eq_mat = (mat.T / v).dot(mat)
    eq_vec = (mat.T / v).dot(obs)
    if gprior is not None:
        eq_mat += np.diag(1.0 / gprior[1] ** 2)
        eq_vec += gprior[0] / gprior[1] ** 2
    return np.linalg.solve(eq_mat, eq_vec)


def solve_ls_b(
    mat: np.ndarray,
    obs: np.ndarray,
    obs_se: np.ndarray,
    bounds: np.ndarray,
    gprior: np.ndarray = None,
    options: Dict = None,
    return_info: bool = False,
) -> Union[np.ndarray, OptimizeResult]:
    """Solve least square with bounds problem

    Args:
        mat(np.ndarray): Data matrix
        obs(np.ndarray): Observations
        obs_se(np.ndarray): Observation standard error.
        bounds(np.ndarray): Bounds for the variable.
        gprior(np.ndarray): Gaussian prior. Default to ``None``.
        options(Dict, optional): Options for scipy solver. Default to None.
        return_info(bool, optional):
            If ``True``, return the convergence information.
            Default to ``False``.

    Returns:
        np.ndarray: Solution.
    """
    x_init = solve_ls(mat, obs, obs_se)
    v = obs_se**2

    if gprior is None:

        def objective(x):
            r = obs - mat.dot(x)
            return 0.5 * np.sum(r**2 / v)

        def gradient(x):
            r = obs - mat.dot(x)
            return (mat.T / v).dot(r)

    else:

        def objective(x):
            r = obs - mat.dot(x)
            return 0.5 * np.sum(r**2 / v) + 0.5 * np.sum(
                (x - gprior[0]) ** 2 / gprior[1] ** 2
            )

        def gradient(x):
            r = obs - mat.dot(x)
            return -(mat.T / v).dot(r) + (x - gprior[0]) / gprior[1] ** 2

    opt_result = minimize(
        objective,
        x0=x_init,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options=options,
    )

    result = opt_result if return_info else opt_result.x

    return result


def result_to_df(
    model,
    data,
    path: str = None,
    prediction: str = "prediction",
    residual: str = "residual",
) -> pd.DataFrame:
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
