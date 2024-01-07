import numpy as np
import pandas as pd
import data_preprocessing as dp
from build import rnn, elm, stat_analysis as sa
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
tickers = pd.read_csv("Forex_ticker.csv", index_col=0)
last_sign = {}


class ArmaGarchStrategy:

    def __init__(self, data: pd.DataFrame, ticker: str, parameters: pd.DataFrame):
        self.data = data
        self.ticker = ticker
        self.parameters = parameters
        self.name = "ArimaGarch"
        self.mod_data = data
        self.mod_data["close"] = self.mod_data["close"] * int(self.parameters.loc[self.ticker, "multiplier"])
        self.order = (int(self.parameters.loc[self.ticker, "p"]),
                      int(self.parameters.loc[self.ticker, "d"]),
                      int(self.parameters.loc[self.ticker, "q"]))
        dist = str(self.parameters.loc[self.ticker, "distribution"])
        if dist == "snorm":
            self.distrib = "normal"
        elif dist == "sstd":
            self.distrib = "skewt"
        else:
            raise ValueError(f"Distribution expected value 'snorm' or 'sstd', {dist} get instead.")

    def generate_signals(self):
        """BACKTESTING PURPOSE\n
        Generate simple long/short signals. It returns -1, 0, 1, respectively for short, hold or long signals"""
        signals = pd.DataFrame(index=self.data.index)
        signals["signal"] = 0.0
        last_signal = 0

        for i in range(len(signals.index), len(signals.index)):
            # Forecast for period i+1
            forec = sa.forecasting(data=self.mod_data.loc[:i, "close"], distrib=self.distrib, order=self.order)
            spread = self.data.loc[i, "spread"] * int(tickers.loc[self.ticker, "pip"])

            # Bearish signal
            if self.mod_data.loc[i, "close"] > (float(forec) + spread) and last_signal != -1:
                signals.loc[i, "signal"] = -1
                last_signal = -1

            # Bullish signal
            elif (self.mod_data.loc[i, "close"] + spread) < float(forec) and last_signal != 1:
                signals.loc[i, "signal"] = 1
                last_signal = 1

            # Holding signal
            else:
                signals.loc[i, "signal"] = 0

        signals.loc[len(signals.index) - 1, "signal"] = 0

        return signals

    def generate_weighted_signals(self):
        """BACKTESTING PURPOSE\n
        Extention of the 'generate_signals()' function. Generate signals of different sizes depending
        on the expected return"""
        signals = pd.DataFrame(index=self.data.index)
        signals["signal"] = 0.0
        last_signal = 0

        for i in range(len(signals.index), len(signals.index)):
            # Forecast for period i+1
            forec = sa.forecasting(data=self.mod_data.loc[:i, "close"], distrib=self.distrib, order=self.order)
            spread = self.data.loc[i, "spread"] * int(tickers.loc[self.ticker, "pip"])
            ret_forec = (float(forec) - self.mod_data.loc[i, "close"]) / self.mod_data.loc[i, "close"]

            # Bearish signal
            if self.mod_data.loc[i, "close"] > (float(forec) + spread) and last_signal != -1:
                if ret_forec < -0.005:
                    signals.loc[i, "signal"] = -2

                else:
                    signals.loc[i, "signal"] = -1

                last_signal = -1

            # Bullish signal
            elif (self.mod_data.loc[i, "close"] + spread) < float(forec) and last_signal != 1:
                if ret_forec > 0.005:
                    signals.loc[i, "signal"] = 2

                else:
                    signals.loc[i, "signal"] = 1

                last_signal = 1

            # Holding signal
            else:
                signals.loc[i, "signal"] = 0

        signals.loc[len(signals.index) - 1, "signal"] = 0

        return signals

    def live_signals(self):
        """For live trading use."""
        global last_sign
        # Forecast for next period
        forec = sa.forecasting(data=self.mod_data.loc[:, "close"], distrib=self.distrib, order=self.order)
        spread = self.data.loc[len(self.mod_data.index) - 1, "spread"] * int(tickers.loc[self.ticker, "pip"])

        # Bearish signal
        if self.mod_data.loc[len(self.mod_data.index) - 1, "close"] > (float(forec) + spread) and \
                last_sign[self.ticker] != 1:
            sign = -1

        # Bullish signal
        elif (self.mod_data.loc[len(self.mod_data.index) - 1, "close"] + spread) < float(forec) and \
                last_sign[self.ticker] != 0:
            sign = 1

        # Holding signal
        else:
            sign = 0

        return sign


class RNNStrategy:

    def __init__(self, data: pd.DataFrame, ticker: str, model):
        self.data = data
        self.ticker = ticker
        self.model = model
        self.name = "RNN"
        self.mod_data = dp.preprocessing(self.data)

    def generate_signals(self):
        """BACKTESTING PURPOSE\n
                Generate simple long/short signals. It returns -1, 0, 1, respectively for short, hold or long signals"""
        signals = pd.DataFrame(index=self.data.index.to_list()[64:])
        signals["signal"] = 0.0
        last_signal = 0

        for i in signals.index.to_list():
            # Forecast for period i+1
            x = self.mod_data.loc[i - 64: i-1].to_numpy()
            x = x.reshape((1, 64, 8))
            pred = rnn.prediction(model=self.model, x=x)

            # Bearish signal
            if pred < 0 and last_signal != -1:
                signals.loc[i, "signal"] = -1
                last_signal = -1

            # Bullish signal
            elif pred > 0 and last_signal != 1:
                signals.loc[i, "signal"] = 1
                last_signal = -1

            # Holding signal
            else:
                signals.loc[i, "signal"] = 0

            return signals

    def live_signals(self):
        """For live trading use."""
        global last_sign
        x = self.mod_data.iloc[-64:].to_numpy()
        x = x.reshape((1, 64, 8))
        pred = rnn.prediction(model=self.model, x=x)

        # Bearish signal
        if pred < 0 and last_sign[self.ticker] != 1:
            sign = -1

        # Bullish signal
        elif pred > 0 and last_sign[self.ticker] != 0:
            sign = 1

        # Holding signal
        else:
            sign = 0

        return sign


class ELMStrategy:

    def __init__(self, data: pd.DataFrame, ticker: str, beta: np.ndarray = None,
                 days: int = 10, timeframe: str = "M", live: bool = False, save: bool = False):
        """Simple trading strategy that uses an Extreme learning machine to make one-step-ahead price predictions.\n
        Parameters:\n
        data: DataFrame containing OHLCV data\n
        ticker: Ticker of the currency cross used\n
        beta: If given, it's the trained weight matrix\n
        days: Number of days considered in the training process\n
        timeframe: Timeframe of the price series used, string between D, H, M, respectively for daily, hourly and 5
        minutes data\n
        live: Boolean value; if True, the last row - aka the actual price - is returned separetly to make
        predictions."""
        self.data = data
        self.ticker = ticker
        self.name = "ELM"
        if live:
            self.mod_data, self.Y, self.last = dp.elm_preprocessing(self.data.copy(deep=True), days=days,
                                                                    timeframe=timeframe, live=True)
        else:
            self.mod_data, self.Y = dp.elm_preprocessing(self.data.copy(deep=True), days=days, timeframe=timeframe)
        self.mod_arr = np.array(self.mod_data)
        if timeframe == "D":
            mul = 4
        elif timeframe == "H":
            mul = 24 * 4
        elif timeframe == "M":
            mul = 12 * 24 * 4
        else:
            raise ValueError(f"String between 'D', 'H' or 'M' expected, got {timeframe} instead.")

        self.model = elm.ELM(num_input_nodes=days * mul,
                             num_hidden_units=days * mul * 10,
                             num_out_units=1,
                             activation="tanh",
                             loss="mse",
                             beta_init=beta,
                             w_init="xavier",
                             verbose=False
                             )

        if not isinstance(beta, np.ndarray):
            self.model.fit(x=self.mod_arr, y=self.Y, display_time=True)

        if save:
            pd.DataFrame(self.model.beta).to_csv(f"{ROOT_DIR}\\ElmBetas\\{timeframe}\\{ticker}_Beta.csv")

    def generate_signals(self):
        """BACKTESTING PURPOSE\n
                Generate simple long/short signals. It returns -1, 0, 1, respectively for short, hold or long signals"""
        signals = pd.DataFrame(index=self.mod_data.index)
        signals["signal"] = 0.0
        last_signal = 0
        for idx, row in enumerate(self.mod_arr):
            time_idx = self.mod_data.index[idx]
            forec = self.model(row)
            spread = self.data.loc[time_idx, "spread"] * float(tickers.loc[self.ticker, "pip"])

            # Bearish signal
            if (forec + spread) < row[3] and last_signal != -1:
                signals.loc[time_idx, "signal"] = -1
                last_signal = -1

            # Bullish signal
            elif forec > (row[3] + spread) and last_signal != 1:
                signals.loc[time_idx, "signal"] = 1
                last_signal = 1

            # Holding signal
            else:
                signals.loc[time_idx, "signal"] = 0
        return signals

    def live_signals(self, x):
        """For live trading use."""
        global last_sign
        pred = self.model(x)
        spread = self.data["spread"].iloc[-1] * float(tickers.loc[self.ticker, "pip"])

        # Bearish signal
        if (pred + spread) < self.mod_data["close"].iloc[-1] and last_sign[self.ticker] != 1:
            sign = -1

        # Bullish signal
        elif pred > (self.mod_data["close"].iloc[-1] + spread) and last_sign[self.ticker] != 0:
            sign = 1

        # Holding signal
        else:
            sign = 0

        return sign
