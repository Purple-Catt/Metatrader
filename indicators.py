import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts


def bollinger_bands(data: pd.DataFrame, period: int = 20, multiplier: float = 2.0):
    sma = data["close"].rolling(window=period).mean()
    std = data["close"].rolling(window=period).std()
    upper_band = sma + (multiplier * std)
    lower_band = sma - (multiplier * std)
    upper_band.name = "UpBB_Band"
    lower_band.name = "LowBB_Band"
    return pd.concat([upper_band, lower_band], axis=1)


def stochastic_oscillator(data: pd.DataFrame, k: int = 10, d: int = 3, slow_factor: int = 6):
    faststoch = ((data["close"] - data["low"].rolling(window=k).min()) / (data["high"].rolling(window=k).max()
                                                                          - data["low"].rolling(window=k).min())) * 100
    slowstoch = faststoch.rolling(window=d).mean()
    faststoch = faststoch.rolling(window=slow_factor).mean()
    slowstoch = slowstoch.rolling(window=slow_factor).mean()
    faststoch.name = "FastStoch"
    slowstoch.name = "SlowStoch"

    return pd.concat([faststoch, slowstoch], axis=1)


def macd(data: pd.DataFrame, slow: int = 26, fast: int = 12, slow_factor: int = 9):
    macd_1 = data["close"].ewm(span=fast).mean() - data["close"].ewm(span=slow).mean()
    macd_sl = macd_1.ewm(span=slow_factor).mean()
    macd_1.name = "MACD"
    macd_sl.name = "MACD_SL"
    return pd.concat([macd_1, macd_sl], axis=1)


def avg_cross(data: pd.DataFrame, slow: int = 50, fast: int = 9):
    slow_ma = data["close"].ewm(span=slow).mean()
    fast_ma = data["close"].ewm(span=fast).mean()
    slow_ma.name = "EMA_Slow"
    fast_ma.name = "EMA_Fast"
    return pd.concat([slow_ma, fast_ma], axis=1)


def rsi(data: pd.DataFrame, period: int = 14):
    close_delta = data["close"].diff()
    up = close_delta.clip(lower=0)
    down = - 1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    ma_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rsindex = ma_up / ma_down
    rsindex = 100 - (100/(1 + rsindex))
    rsindex.name = "RSI"
    return rsindex


def timeseries_test(data: pd.Series, alpha=0.05):
    # Augmented Dickey-Fuller test
    test_1 = ts.adfuller(data, 1)
    # Hurst exponent
    listed_values = data.to_list()
    lags = range(2, 50)
    tau = [np.std(np.subtract(listed_values[lag:], listed_values[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return test_1[1] < alpha, poly[0]
