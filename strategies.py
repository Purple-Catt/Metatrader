import numpy as np
import pandas as pd
import stat_analysis as sa
import data_preprocessing as dp
import RNN
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
            raise ValueError(f"Distribution expected value 'snorm' or 'sstd', {dist} found instead.")

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

    def live_signals(self):
        """For live trading use."""
        global last_sign
        x = self.mod_data.iloc[-64:].to_numpy()
        x = x.reshape((1, 64, 8))
        pred = RNN.prediction(model=self.model, x=x)

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
