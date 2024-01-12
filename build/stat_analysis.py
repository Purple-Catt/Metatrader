import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import arch
import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parameters = pd.read_csv(f"{ROOT_DIR}\\Data\\Params\\Parameter_daily.csv", index_col=0)
parameters_hourly = pd.read_csv(f"{ROOT_DIR}\\Data\\Params\\Parameters_hourly.csv", index_col=0)


def adf_test(data: pd.DataFrame, alpha=0.05):
    """AD-Fuller test for stationarity."""
    result = adfuller(data["close"])
    if result[1] > alpha:
        # Time series is not stationary
        return 1
    else:
        # Time series is stationary
        return 0


def best_arima_model(data, d):
    """Best parameters determination of an ARIMA model based on lowest AIC. Return a list useful for model fitting."""
    stepwise_model = pm.auto_arima(data, start_p=0, d=d, start_q=0, max_p=4, max_q=4, seasonal=False)
    return stepwise_model.get_params()["order"]


"""def r_bestmodel_arimagarch(pydata, distrib: list):
    # Import libraries
    base = importr("base")
    bar = importr("BestArimaGarch")
    # Convert Python dataframe to R array
    with ro.default_converter + pandas2ri.converter:
        rdata = ro.conversion.get_conversion().py2rpy(pydata)
    cond = ro.vectors.StrVector(distrib)
    # Run R script
    bar.garchAuto(base.diff(rdata), min_order=ro.vectors.IntVector([0, 0, 1, 1]),
                  max_order=ro.vectors.IntVector([2, 2, 1, 1]), trace=True, cond_dists=cond, with_forecast=False)
"""


def forecasting(data, distrib, order):
    """Forecast the return of one-step-ahead period using an ARIMA-GARCH model."""
    arimamodel = ARIMA(data, order=order)
    arimafit = arimamodel.fit()
    resids = arimafit.resid
    pred_mu = arimafit.forecast(steps=1)
    archmodel = arch.arch_model(resids, p=1, q=1, vol="GARCH", dist=distrib, rescale=False)
    archfit = archmodel.fit()
    forc = archfit.forecast(horizon=1)
    pred_et = forc.mean["h.1"].iloc[-1]
    prediction = pred_mu + pred_et
    return prediction


def rkv(data: pd.DataFrame, ret: pd.DataFrame):
    # TODO
    """Implementation of Realized Kernels, from Realized Kernels in practice, Barndorff-Nielsen et al. 2008b.
    Read the paper for a deep explanation at the following link:\n
    http://public.econ.duke.edu/~get/browse/courses/201/spr12/DOWNLOADS/MicroStructure/bhls_kernels_practice_08.pdf"""
    # Defining parzen kernel function
    def parzen_kernel(x):
        if 0.0 <= x <= 0.5:
            k = 1 - (6*(x**2)) + (6*(x**3))
        elif 0.5 <= x <= 1:
            k = 2 * ((1-x)**3)
        elif x > 1:
            k = 0
        else:
            raise ValueError("x is negative. A positive value is mandatory")
        return k

    def gamma(h, ret: list):
        n = len(ret)
        prod = []
        for j in range(abs(h)+1, n+1):
            prod.append(ret[j] * ret[j - abs(h)])
        return np.sum(prod)

    # Optimal bandwidth (Barndorff-Nielsen et al.)
    def h_upper():
        c_star = ((12**2) / 0.269)**(1 / 5)
        rv_dict = {}
        for n in range(1200):
            ret_sq = []
            for j in range(1+n, len(ret["bid"]), 20):
                ret_sq.append((ret.iloc[j, ret.columns.get_loc("bid")]) ** 2)
            rv_dict[n] = np.sum(ret_sq)
        iv = np.mean(rv_dict.values())
        h_star = 0
        return h_star

    partial = []
    for h in range(-h_upper(), h_upper() + 1):
        kernel = parzen_kernel(h / (h_upper() + 1))
        partial.append(kernel * gamma(h=h, ret=list(data.loc[:, "close"].diff().dropna())))
    return np.sum(partial)
