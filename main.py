# IMPORTING LIBRARIES
from datetime import datetime
import sys
import MetaTrader5 as Mt5
import warnings
from keras.models import load_model
import indicators as ind
import stat_analysis as sa
import strategies
from strategies import ArmaGarchStrategy, RNNStrategy
import pytz
# import matplotlib.pyplot as plt
import telegram_bot as tb
import credentials
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
warnings.simplefilter("ignore", FutureWarning)

# Broker redentials
server = credentials.server
real = credentials.real
demo = credentials.demo
# Variables
data_dict = {}  # Time series
pos_dict = {}  # Opened positions
sign_dict = {}  # Signals for backtestings
models = {}  # Models for NN strategies
last_sign = strategies.last_sign
tickers = pd.read_csv("Forex_ticker.csv", index_col=0)
daily = False
timezone = pytz.timezone("Etc/UTC")
# Arima-Garch model parameters
if daily:
    params = sa.parameters
    timeframe = Mt5.TIMEFRAME_D1

else:
    params = sa.parameters_hourly
    timeframe = Mt5.TIMEFRAME_H1

candles = 100  # Number of OHLCV data to download
win = 0  # Number of winning positions
backtest = False  # Set True for backtesting, False for Live trading


def metatrader_start(username, psw, svr):
    """Establish connection to the MetaTrader 5 terminal"""
    if not Mt5.initialize():
        print("initialize() failed, error code =", Mt5.last_error())
        quit()
    authorized = Mt5.login(username, password=psw, server=svr)
    if authorized:
        print("Connection established")
    else:
        print(f"failed to connect at account #{username}, error code: {Mt5.last_error()}")


def time_series_download(n_candles, online: bool = True):
    global data_dict, timeframe
    data_dict = {}

    if online:
        for name in tickers.index:
            try:
                rates = Mt5.copy_rates_from_pos(name, timeframe, 0, n_candles)
                rates_frame = pd.DataFrame(rates)
                rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
                data_dict[name] = rates_frame
                print(f"{name} downloaded correctly")

            except KeyError:
                print(f"Error in downloading {name}")

    else:
        for name in tickers.index:
            if timeframe == Mt5.TIMEFRAME_D1:
                data_dict[name] = pd.read_csv(f"Time series\\Daily\\{name}.csv")

            elif timeframe == Mt5.TIMEFRAME_H1:
                data_dict[name] = pd.read_csv(f"Time series\\Hourly\\{name}.csv")

            else:
                raise ValueError("Timeframe not yet implemented.")


def time_series_saving(folder: str = "Daily"):
    """Save time series into a project folder."""
    global data_dict
    for name in data_dict.keys():
        data_dict[name].to_csv(f"Time series\\{folder}\\{name}.csv")
        print(f"{name} saved correctly")


def signals_loading():
    """ONLY FOR DEVELOPING PURPOSES\n
    Used if signals are locally saved."""
    for name in tickers.index:
        data_dict[name] = pd.read_csv(f"Time series\\Daily\\{name}.csv")
        if params.loc[name, "distribution"] != "0":
            sign_dict[name] = pd.read_csv(f"Signals\\{name} signals.csv")
            print(f"{name} signals and data loaded correctly")


def best_params():
    """Find the best p, d, q params for an ARIMA-GARCH model based on lowest AIC
    saving results in console_output file."""
    sys.stdout = open("console_output.txt", "w")
    for i in tickers.index:
        if params.loc[i, "distribution"] == "0":
            pass
        else:
            distr = str(params.loc[i, "distribution"])
            data = data_dict[i].loc[:, "close"] * int(params.loc[i, "multiplier"])
            sa.r_bestmodel_arimagarch(data, list(distr))
    sys.stdout.close()


def market_order(symbol: str, kind: str, act: str, lot: int = 0.1):
    """Send and execute a real trade on MetaTrader5. A response will be visible via console and Telegram channel."""
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
            tb.send_message(f"*{kind.title()}* order for *{lot}* lots on _{symbol}_ failed.")

        else:
            print("Execution completed succesfully. Position opened.\n")
            tb.send_message(f"*{kind.title()}* order for *{lot}* lots on _{symbol}_ executed succesfully.")

    elif act == "close":

        if kind == "short":
            types = Mt5.ORDER_TYPE_BUY

        else:
            types = Mt5.ORDER_TYPE_SELL
        order = Mt5.positions_get(symbol=symbol)[0].ticket
        request = {
            "action": Mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": types,
            "position": order,
            "type_filling": Mt5.ORDER_FILLING_FOK,
            "type_time": Mt5.ORDER_TIME_DAY
        }

        result = Mt5.order_send(request)
        profit = Mt5.history_deals_get(ticket=result.deal)[0].profit
        print(f"Closing order on position {order} for {lot} lots on {symbol} sent.")

        if result.retcode != Mt5.TRADE_RETCODE_DONE:
            print(f"Execution failed. Retcode: {result.retcode}")
            tb.send_message(f"Position *{order}* failed to close.")

        else:
            print("Execution completed succesfully. Position closed.\n")
            tb.send_message(f"Position *{order}* closed succesfully.\n*Profit*: {profit}")


class Position:
    """When a trade is done, a Position object is created. It contains all the useful
    functions and properties of the trade itself."""
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

    def isopen(self):
        """This function shows if the position is still open or not."""
        if self.closeind == 0:
            return True
        else:
            return False

    def iswinner(self):
        """FOR ANALYTICAL PURPOSES\n
        It shows if the trade was in profit(loss)."""
        if self.earnings() > 0:
            return True
        else:
            return False

    def buy_cashflow(self):
        """Euros spent to execute the operation."""
        cf = self.qty / self.open[1]
        return cf

    def sl_price(self):
        """Return the SL price, given an %SL."""
        if self.kind == "long":
            price = self.open[0] * (1 - self.sl)
        else:
            price = self.open[0] * (1 + self.sl)
        return price

    def getkind(self):
        """It returns the kind (stop/loss) of trade."""
        return self.kind

    def getclose(self):
        """It returns the index of the closing price."""
        return self.closeind

    def getopen(self):
        """It returns the opening price."""
        return self.open[0]

    def setclose(self, close: float, closeexc: int, closeind: int):
        """Used to close the trade."""
        self.close = [close, closeexc]
        self.closeind = closeind


class Portfolio:
    """This class creates an actual portfolio for one financial instrument and generates the positions, given a
    signal DataFrame. It's used in backtesting."""
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
        """Open long (short) positions based on signals and eventually close already opened ones."""
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.ticker] = 0.0
        positions["buycf"] = 0.0
        positions["sellcf"] = 0.0
        for i in self.signals.index:

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
    # ARIMA-GARCH STRATEGY
    # BACKTESTING
    if backtest:
        time_series_download(candles, online=False)
        bt_optimization = pd.DataFrame(index=tickers.index)
        for sl in [0.025, 0.05, 0.1]:
            bt_return = pd.DataFrame(index=tickers.index)
            for tkr in tickers.index:
                max_dd = 0.0
                if params.loc[tkr, "distribution"] == "0":
                    pass
                else:
                    pos_dict = {}
                    strategy = ArmaGarchStrategy(data_dict[tkr], ticker=tkr, parameters=params)
                    win = 0
                    returns = Portfolio(data=data_dict,
                                        ticker=tkr,
                                        exchange=tickers.loc[tkr, "Toeuro"],
                                        signals=strategy.generate_weighted_signals(),
                                        stoploss=sl).backtest_portfolio()
                    returns.to_csv(f"Portfolios\\{strategy.name}\\{tkr}.csv")
                    bt_return.loc[tkr, strategy.name] = returns.loc[candles - 1, "cash"]
                    for n in pos_dict.values():
                        # Number of winning positions
                        if n.iswinner() is True:
                            win += 1
                        # Maximum drawdown
                        if n.earnings() < max_dd:
                            max_dd = n.earnings()
                    bt_return.loc[tkr, f"{strategy.name} % Win"] = (win / len(pos_dict.keys())) * 100
                    bt_return.loc[tkr, f"{strategy.name} Win"] = win
                    bt_return.loc[tkr, f"{strategy.name}  Operations"] = len(pos_dict)
                    bt_return.loc[tkr, "Max DD"] = max_dd
                    print(f"{sl} {strategy.name} strategy on {tkr} successfully tested")
                    bt_optimization.loc[tkr, f"Ret {sl}"] = returns.loc[candles - 1, "cash"] - 100000
                    bt_optimization.loc[tkr, f"MaxDD {sl}"] = max_dd
        bt_optimization.to_csv("Opt_sl.csv")

    # LIVE TRADING
    else:
        tkr_list = []
        metatrader_start(demo[0], demo[1], demo[2])
        for tkr in tickers.index:
            # if params.loc[tkr, "distribution"] != "0":
            tkr_list.append(tkr)  # Used for ArmaGarch strategy
            models[tkr] = load_model(f"Models\\{tkr}.keras")
            # Find and save already opened trade on the same ticker
            try:
                last_sign[tkr] = Mt5.positions_get(symbol=tkr)[0].type
            except AttributeError:
                last_sign[tkr] = -1
            except IndexError:
                last_sign[tkr] = -1
        Mt5.shutdown()
        x = True
        while True:
            # For a multiday trading, every night before the end of the trading day, the process begin
            now = datetime.now(timezone).strftime("%M:%S")
            if now == "55:00":
            # if x:
                metatrader_start(demo[0], demo[1], demo[2])
                time_series_download(candles)
                for tkr in tkr_list:
                    try:
                        last_sign[tkr] = Mt5.positions_get(symbol=tkr)[0].type
                    except AttributeError:
                        last_sign[tkr] = -1
                    except IndexError:
                        last_sign[tkr] = -1
                    # Define the strategy to use
                    '''
                    strategy = ArmaGarchStrategy(data_dict[tkr],
                                                 ticker=tkr,
                                                 parameters=params)
                    '''
                    strategy = RNNStrategy(data=data_dict[tkr],
                                           ticker=tkr,
                                           model=models[tkr])
                    act_signal = strategy.live_signals()
                    # Open long position
                    if act_signal > 0:
                        opened = Mt5.positions_get(symbol=tkr)

                        if opened is not None:
                            # Close already opened position
                            for op in opened:
                                market_order(symbol=tkr, kind="short", act="close")

                        market_order(symbol=tkr, kind="long", act="open")

                    # Open short position
                    elif act_signal < 0:
                        opened = Mt5.positions_get(symbol=tkr)

                        if opened is not None:
                            # Close already opened position
                            for op in opened:
                                market_order(symbol=tkr, kind="long", act="close")

                            market_order(symbol=tkr, kind="short", act="open")

                Mt5.shutdown()
                x = False
