# IMPORTING LIBRARIES
from datetime import datetime
import sys
import MetaTrader5 as Mt5
import indicators as Ind
import stat_analysis as sa
import pytz
# import matplotlib.pyplot as plt
import credentials
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Credentials
server = credentials.server
real = credentials.real
demo = credentials.demo
# Variables
data_dict = {}  # Time series are going to be loaded here
sec_data = {}  # A modified time series is going to be loaded here
ret_data = {}  # Returns are going to be loaded here
pos_dict = {}  # Opened positions are going to be loaded here
sign_dict = {}  # Signals for backtesting are going to be loaded here
tickers = pd.read_csv("Forex_ticker.csv", index_col=0)
daily = False
# Arima-Garch model parameters
if daily is True:
    params = sa.parameters
else:
    params = sa.parameters_hourly
# Datetime info to avoid downloading errors
timezone = pytz.timezone("Etc/UTC")
timeframe = Mt5.TIMEFRAME_H1  # Candles timeframe
candles = 90000  # Number of OHLCV to download
win = 0  # Number of winning positions are going to be saved here
last_sign = {}  # Last signal is going to be loaded here
backtest = True  # Set True for backtesting, False for Live trading


# Connection to the MT5 account
def metatrader_start(username, psw, svr):
    print("MetaTrader5 package author: ", Mt5.__author__)
    print("MetaTrader5 package version: ", Mt5.__version__)
    # Establish connection to the MetaTrader 5 terminal
    if not Mt5.initialize():
        print("initialize() failed, error code =", Mt5.last_error())
        quit()
    # Display data on MetaTrader 5 version
    print(Mt5.version())
    authorized = Mt5.login(username, password=psw, server=svr)
    if authorized:
        print("Connected established")
    else:
        print(f"failed to connect at account #{username}, error code: {Mt5.last_error()}")


def time_series_download(n_candles):
    global data_dict, timeframe
    data_dict = {}
    for name in tickers.index:
        try:
            rates = Mt5.copy_rates_from_pos(name, timeframe, 0, n_candles)
            rates_frame = pd.DataFrame(rates)
            rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
            data_dict[name] = rates_frame
            print(f"{name} downloaded correctly")
        except KeyError:
            print(f"Error in downloading {name}")


# Eventually used to save Time Series into a project folder
def time_series_saving():
    global data_dict
    for name in data_dict.keys():
        data_dict[name].to_csv(f"Time series\\Hourly\\{name}.csv")
        print(f"{name} saved correctly")


"""
MANDATORY
the 'data_cleaning' function download 24h ticks data from MT5 and clean them as showed in 
'Realized Kernels in Practice: Trades and Quotes Barndorff-Nielsen et al. 2008b'

It must be used for the RV strategy, that still need to be implemented in this code
"""


def data_cleaning():
    # Download time series
    for name in tickers.index:
        try:
            # Define the time range
            date_from = datetime(2023, 5, 8, 23, 59, 59, tzinfo=timezone)
            date_to = datetime(2023, 5, 10, 0, 0, 0, tzinfo=timezone)
            # Download time series
            rates = Mt5.copy_ticks_range(name, date_from, date_to, Mt5.COPY_TICKS_ALL)
            tickframe = pd.DataFrame(rates)
            print(f"{name} downloaded correctly")
            # Remove unused columns
            tickframe["time"] = pd.to_datetime(tickframe["time"], unit="s")
            tickframe.drop(["last", "volume", "time_msc", "flags", "volume_real"], axis=1, inplace=True)
            data_dict[name] = tickframe
            sec_index = pd.date_range(start=date_from, end=date_to, freq="S", tz=timezone, inclusive="left")
            sec_data[name] = pd.DataFrame(np.NaN, index=sec_index, columns=["bid", "ask"])
            # Drop rows where bid and/or ask price is null
            data_dict[name].drop(data_dict[name].loc[data_dict[name]["bid"] == 0.0].index, inplace=True)
            data_dict[name].drop(data_dict[name].loc[data_dict[name]["ask"] == 0.0].index, inplace=True)
            # Drop rows where bid/ask spread is negative
            spread = data_dict[name]["ask"] - data_dict[name]["bid"]
            data_dict[name].drop(data_dict[name].loc[(spread < 0), :].index, inplace=True)
            # Replace quotes with the same timestamp with a single entry with the median bid/ask price
            vc = tickframe["time"].value_counts()
            dupl = vc[vc > 1].index
            for i in dupl:
                filt = tickframe.loc[tickframe["time"] == i, ["bid", "ask"]]
                bid_median = round(filt["bid"].median(), int(tickers.loc[name, "dec"]))
                ask_median = round(filt["ask"].median(), int(tickers.loc[name, "dec"]))
                bid_last = filt.iloc[len(filt["bid"]) - 1, filt.columns.get_loc("bid")]
                ask_last = filt.iloc[len(filt["ask"]) - 1, filt.columns.get_loc("ask")]
                index = list(filt.index)
                data_dict[name].loc[index[0], "bid"] = bid_median
                data_dict[name].loc[index[0], "ask"] = ask_median
                sec_data[name].loc[i, "bid"] = bid_last
                sec_data[name].loc[i, "ask"] = ask_last
                index = index[1:]
                data_dict[name].drop(index=index, inplace=True)
            # Retrieve already known values
            for i in data_dict[name].index:
                date = data_dict[name].loc[i, "time"]
                if sec_data[name].loc[date, "bid"] == np.NaN:
                    sec_data[name].loc[date, "bid"] = data_dict[name].loc[i, "bid"]
                    sec_data[name].loc[date, "ask"] = data_dict[name].loc[i, "ask"]
                else:
                    pass
            # Insert missing values
            prev = None
            for i in sec_data[name].index:
                if sec_data[name].loc[i, "bid"] == np.NaN:
                    if prev is None:
                        pass
                    else:
                        sec_data[name].loc[i, "bid"] = sec_data[name].loc[prev, "bid"]
                        sec_data[name].loc[i, "ask"] = sec_data[name].loc[prev, "ask"]
                else:
                    pass
                prev = i
            ret_data[name] = sec_data["bid"].diff()
            ret_data[name].dropna(inplace=True)
            sec_data[name].drop(sec_data[name].index[0], inplace=True)
            data_dict[name].drop(data_dict[name].index[0], inplace=True)
            print(ret_data[name])
            print(f"{name} data cleaned")
        except KeyError:
            print(f"Error in downloading {name}")


# ONLY FOR DEVELOPING PURPOSES
# Used if signals are locally saved
def signals_loading():
    for name in tickers.index:
        data_dict[name] = pd.read_csv(f"Time series\\Daily\\{name}.csv")
        if params.loc[name, "distribution"] != "0":
            sign_dict[name] = pd.read_csv(f"Signals\\{name} signals.csv")
            print(f"{name} signals and data loaded correctly")


# UNUSED
# If some strategies required them, this function calculate some indicators
def indicators():
    global data_dict
    for name in data_dict.keys():
        datas = data_dict[name]
        data_dict[name] = pd.concat([data_dict[name], Ind.bollinger_bands(datas), Ind.stochastic_oscillator(datas),
                                     Ind.macd(datas), Ind.avg_cross(datas), Ind.rsi(datas)], axis=1)


# Find the best p, d, q params for an ARIMA-GARCH model based on lowest AIC saving results in console_output file
def best_params():
    sys.stdout = open("console_output.txt", "w")
    for i in tickers.index:
        if params.loc[i, "distribution"] == 0:
            pass
        else:
            distr = str(params.loc[i, "distribution"])
            sa.r_bestmodel_arimagarch(data_dict[i].loc[:, "close"], list(distr))
    sys.stdout.close()


# FOR LIVE TRADING USE
# Execute a real trade on MetaTrader5
def market_order(symbol: str, kind: str, act: str, lot: int = 0.02):
    if act == "open":
        if kind == "long":
            types = Mt5.ORDER_TYPE_BUY
        else:
            types = Mt5.ORDER_TYPE_SELL
        request = {
            "action": Mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": types,
            "type_filling": Mt5.ORDER_FILLING_FOK,
            "type_time": Mt5.ORDER_TIME_DAY
        }
        result = Mt5.order_send(request)
        print(f"{kind.title()} order for {lot} lots on {symbol} sent.")
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            print(f"Execution failed. Retcode: {result.retcode}")
        else:
            print("Execution completed succesfully. Position opened.\n")
    elif act == "close":
        if kind == "short":
            types = Mt5.ORDER_TYPE_BUY
        else:
            types = Mt5.ORDER_TYPE_SELL
        order = Mt5.positions_get(symbol).ticket
        request = {
            "action": Mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": types,
            "position": order,
            "type_filling": Mt5.ORDER_FILLING_RETURN,
            "type_time": Mt5.ORDER_TIME_DAY
        }
        result = Mt5.order_send(request)
        print(f"Closing order on position {order} for {lot} lots on {symbol} sent.")
        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            print(f"Execution failed. Retcode: {result.retcode}")
        else:
            print("Execution completed succesfully. Position closed.\n")


# When a trade is done, a Position object is created. It contains all the useful
# functions and properties of the trade itself
class Position:
    def __init__(self, ticker, exchange: str, kind: str, opens: list,
                 close=None, close_index=0, stoploss=0.015, quantity=50000):
        # open and close are lists that contain the closing prices of pair and exchange
        if close is None:
            close = [0.0, 0]
        self.tkr = ticker
        self.exchange = exchange  # NOT USED
        self.kind = kind
        self.open = opens
        self.close = close
        self.closeind = close_index
        self.sl = stoploss
        self.qty = quantity

    def earnings(self):
        profit = 0.0
        if self.kind == "long":
            # P&L in quote currency
            pel = (self.close[0] - self.open[0]) * self.qty
            # P/L in Euro
            profit = pel / self.close[1]
        elif self.kind == "short":
            # P&L in quote currency
            pel = (self.open[0] - self.close[0]) * self.qty
            # P/L in Euro
            profit = pel / self.close[1]
        return profit

    # This function shows if the position is still open or not
    def isopen(self):
        if self.closeind == 0:
            return True
        else:
            return False

    # It shows if the trade was in profit(loss) FOR ANALYTICAL PURPOSES
    def iswinner(self):
        if self.earnings() > 0:
            return True
        else:
            return False

    # Euros spent to execute the operation
    def buy_cashflow(self):
        cf = self.qty / self.open[1]
        return cf

    # Return the SL price, given an %SL
    def sl_price(self):
        if self.kind == "long":
            price = self.open[0] * (1 - self.sl)
        else:
            price = self.open[0] * (1 + self.sl)
        return price

    # It returns the kind (stop/loss) of trade
    def getkind(self):
        return self.kind

    # It returns the index of the closing price
    def getclose(self):
        return self.closeind

    # It returns the opening price
    def getopen(self):
        return self.open[0]

    # Used to close the trade
    def setclose(self, close: float, closeexc: int, closeind: int):
        self.close = [close, closeexc]
        self.closeind = closeind


class ArmaGarchStrategy:
    def __init__(self, data: pd.DataFrame, ticker: str, order: tuple, distrib: str):
        self.data = data
        self.ticker = ticker
        self.order = order
        self.distrib = distrib
        self.mod_data = data
        self.mod_data["close"] = self.mod_data["close"] * int(params.loc[self.ticker, "multiplier"])

    # BACKTESTING PURPOSE
    # Generate simple long/short signals
    def generate_signals(self):
        # Return -1, 0, 1, respectively for short, hold or long signals
        signals = pd.DataFrame(index=self.data.index)
        signals["signal"] = 0.0
        last_signal = 0
        for i in range(len(signals.index) - 240, len(signals.index)):
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

    # BACKTESTING PURPOSE
    # Generate signals of different sizes depending on the expected return
    def generate_weighted_signals(self):
        signals = pd.DataFrame(index=self.data.index)
        signals["signal"] = 0.0
        last_signal = 0
        for i in range(len(signals.index) - 240, len(signals.index)):
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

    # FOR LIVE TRADING
    def live_signals(self):
        global last_sign
        # Forecast for next period
        forec = sa.forecasting(data=self.mod_data.loc[:, "close"], distrib=self.distrib, order=self.order)
        spread = self.data.loc[len(self.mod_data.index) - 1, "spread"] * int(tickers.loc[self.ticker, "pip"])
        # Bearish signal
        if self.mod_data.loc[len(self.mod_data.index) - 1, "close"] > (float(forec) + spread) and \
                last_sign[self.ticker] != 1:
            sign = -1
            last_sign[self.ticker] = 1
        # Bullish signal
        elif (self.mod_data.loc[len(self.mod_data.index) - 1, "close"] + spread) < float(forec) and \
                last_sign[self.ticker] != 0:
            sign = 1
            last_sign[self.ticker] = 0
        # Holding signal
        else:
            sign = 0
        return sign


class Portfolio:
    def __init__(self, data: dict, ticker: str, exchange: str,
                 signals: pd.DataFrame, init_capital=100000.0, stoploss=0.015):
        self.tkr_data = data[ticker]
        self.all_data = data
        self.ticker = ticker
        self.exchange = exchange
        self.signals = signals
        self.init_capital = init_capital
        self.sl = stoploss
        self.positions = self.generate_positions()

    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.ticker] = 0.0
        positions["buycf"] = 0.0
        positions["sellcf"] = 0.0
        for i in self.signals.index:
            # Open long (short) positions based on signals and eventually close already opened ones
            if tickers.loc[self.ticker, "Base"] == "EUR":
                cross = 1.0
            else:
                cross = self.all_data["EUR" + tickers.loc[self.ticker, "Base"]].loc[i, "close"]
            # Determine the quantity
            if abs(self.signals.loc[i, "signal"]) == 1:
                qty = 25000
            elif abs(self.signals.loc[i, "signal"]) == 2:
                qty = 50000
            else:
                qty = 0
            # Bullish signal
            spread = self.tkr_data.loc[i, "spread"] * int(tickers.loc[self.ticker, "pip"])
            if self.signals.loc[i, "signal"] > 0:
                pos_dict[i] = Position(ticker=self.ticker,
                                       exchange=self.exchange,
                                       kind="long",
                                       opens=[self.tkr_data.loc[i, "close"] + spread, cross],
                                       stoploss=self.sl,
                                       quantity=qty)
                # Close opposite positions
                for pos in pos_dict.values():
                    if pos.getkind() == "short" and pos.isopen() is True:
                        pos.setclose(self.tkr_data.loc[i, "close"] + spread,
                                     self.all_data[self.exchange].loc[i, "close"], i)
            # Bearish signal
            elif self.signals.loc[i, "signal"] < 0:
                pos_dict[i] = Position(ticker=self.ticker,
                                       exchange=self.exchange,
                                       kind="short",
                                       opens=[self.tkr_data.loc[i, "close"], cross],
                                       stoploss=self.sl,
                                       quantity=qty)
                # Close opposite positions
                for pos in pos_dict.values():
                    if pos.getkind() == "long" and pos.isopen() is True:
                        pos.setclose(self.tkr_data.loc[i, "close"], self.all_data[self.exchange].loc[i, "close"], i)
            else:
                pass
            # Check stop loss
            for pos in pos_dict.values():
                if pos.isopen() is True \
                        and pos.getkind() == "long" \
                        and pos.sl_price() > self.tkr_data.loc[i, "close"]:
                    pos.setclose(self.tkr_data.loc[i, "close"],
                                 self.all_data[self.exchange].loc[i, "close"], i)
                elif pos.isopen() is True \
                        and pos.getkind() == "short" \
                        and pos.sl_price() < self.tkr_data.loc[i, "close"]:
                    pos.setclose(self.tkr_data.loc[i, "close"] + spread,
                                 self.all_data[self.exchange].loc[i, "close"], i)
        # Close opened positions at the end of the backtest
        for pos in pos_dict.values():
            if pos.isopen() is True:
                pos.setclose(self.tkr_data.loc[self.tkr_data.index[-1], "close"],
                             self.all_data[self.exchange].loc[self.tkr_data.index[-1], "close"],
                             self.tkr_data.index[-1])
        for i in pos_dict.keys():
            positions.loc[i, "buycf"] = - pos_dict[i].buy_cashflow()
            positions.loc[pos_dict[i].getclose(), "sellcf"] = pos_dict[i].buy_cashflow() + pos_dict[i].earnings()
        positions[self.ticker] = positions["buycf"] + positions["sellcf"]
        return positions

    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.signals.index).fillna(0.0)
        portfolio["holdings"] = self.positions[self.ticker]
        portfolio["cash"] = self.init_capital + portfolio["holdings"].cumsum()
        portfolio["total"] = portfolio["cash"] + portfolio["holdings"]
        portfolio["returns"] = portfolio["total"].pct_change() * 100
        self.positions.to_csv(f"POSITION\\{self.ticker} positions.csv")
        return portfolio


if __name__ == "__main__":
    metatrader_start(demo[0], demo[1], demo[2])
    # data_cleaning()
    time_series_download(90000)
    time_series_saving()
    Mt5.shutdown()
    '''
    # ARIMA-GARCH STRATEGY
    # The kind of distribution and p, d, q parameters were find with the best_params function
    # In this case the backtest is used to compare different stop-loss
    if backtest is True:  # BACKTESTING
        bt_optimization = pd.DataFrame(index=tickers.index)
        for sl in [0.025, 0.05, 0.1]:
            bt_return = pd.DataFrame(index=tickers.index)
            for tkr in tickers.index:
                max_dd = 0.0
                if params.loc[tkr, "distribution"] == "0":
                    pass
                else:
                    orders = (int(params.loc[tkr, "p"]), int(params.loc[tkr, "d"]), int(params.loc[tkr, "q"]))
                    dist = str(params.loc[tkr, "distribution"])
                    distribution = ""
                    if dist == "snorm":
                        distribution = "normal"
                    elif dist == "sstd":
                        distribution = "skewt"
                    pos_dict = {}
                    win = 0
                    returns = Portfolio(data=data_dict,
                                        ticker=tkr,
                                        exchange=tickers.loc[tkr, "Toeuro"],
                                        signals=ArmaGarchStrategy(data_dict[tkr],
                                                                  ticker=tkr,
                                                                  order=orders,
                                                                  distrib=distribution).generate_weighted_signals(),
                                        stoploss=sl).backtest_portfolio()
                    returns.to_csv(f"Portfolios\\ArimaGarch\\{tkr}.csv")
                    bt_return.loc[tkr, "ArimaGarch"] = returns.loc[candles - 1, "cash"]
                    for n in pos_dict.values():
                        # Number of winning positions
                        if n.iswinner() is True:
                            win += 1
                        # Maximum drawdown
                        if n.earnings() < max_dd:
                            max_dd = n.earnings()
                    bt_return.loc[tkr, "ArimaGarch % Win"] = (win / len(pos_dict.keys())) * 100
                    bt_return.loc[tkr, "ArimaGarch Win"] = win
                    bt_return.loc[tkr, "ArimaGarch  Operations"] = len(pos_dict)
                    bt_return.loc[tkr, "Max DD"] = max_dd
                    print(f"{sl} ArimaGarch strategy on {tkr} successfully tested")
                    bt_optimization.loc[tkr, f"Ret {sl}"] = returns.loc[candles - 1, "cash"] - 100000
                    bt_optimization.loc[tkr, f"MaxDD {sl}"] = max_dd
        bt_optimization.to_csv("Opt_sl.csv")
    else:  # LIVE TRADING
        tkr_list = []
        for tkr in tickers.index:
            if params.loc[tkr, "distribution"] != "0":
                tkr_list.append(tkr)
                # Find and save already opened trade on the same ticker
                last_sign[tkr] = Mt5.positions_get(tkr).type
            else:
                pass
        while True:
            # For a multiday trading, every night before the end of the trading day, the process begin
            now = datetime.now(timezone).strftime("%H:%M:%S")
            if now == "21:00:00":
                metatrader_start(demo[0], demo[1], demo[2])
                time_series_download(candles)
                for tkr in tkr_list:
                    last_sign[tkr] = Mt5.positions_get(tkr).type
                    orders = (int(params.loc[tkr, "p"]), int(params.loc[tkr, "d"]), int(params.loc[tkr, "q"]))
                    # The following "if" solves a denomination problem between R and Python
                    dist = str(params.loc[tkr, "distribution"])
                    distribution = ""
                    if dist == "snorm":
                        distribution = "normal"
                    elif dist == "sstd":
                        distribution = "skewt"
                    # Define the strategy to use
                    strat = ArmaGarchStrategy(data_dict[tkr],
                                              ticker=tkr,
                                              order=orders,
                                              distrib=distribution)
                    act_signal = strat.live_signals()
                    # Open long position
                    if act_signal > 0:
                        opened = Mt5.orders_get(symbol=tkr)
                        if opened is None:
                            pass
                        else:
                            # Close already opened position
                            for op in opened:
                                market_order(symbol=tkr, kind="short", act="close")
                        market_order(symbol=tkr, kind="long", act="open")
                    # Open short position
                    elif act_signal < 0:
                        opened = Mt5.orders_get(symbol=tkr)
                        if opened is None:
                            pass
                        else:
                            # Close already opened position
                            for op in opened:
                                market_order(symbol=tkr, kind="long", act="close")
                            market_order(symbol=tkr, kind="short", act="open")
                Mt5.shutdown()'''