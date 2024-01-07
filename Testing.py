import MetaTrader5 as Mt5
import warnings
import strategies
from strategies import ELMStrategy
import os
import pytz
import telegram_bot as tb
import credentials
import pandas as pd
import numpy as np
warnings.simplefilter("ignore", FutureWarning)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Broker credentials
server = credentials.server
demo = credentials.demo
# Variables
pos_dict = {}
models = {}  # Models for RNN strategy
last_sign = strategies.last_sign
tickers = pd.read_csv("Forex_ticker.csv", index_col=0)
live = False
timezone = pytz.timezone("Etc/UTC")
timeframe = Mt5.TIMEFRAME_H1

candles = 70000  # Number of OHLCV data to download


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


def time_series_download(n_candles: int = 0, online: bool = True, tf=timeframe):
    data_dict = {}

    if online:
        for name in tickers.index:
            try:
                rates = Mt5.copy_rates_from_pos(name, tf, 0, n_candles)
                rates_frame = pd.DataFrame(rates)
                rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
                data_dict[name] = rates_frame
                print(f"{name} downloaded correctly")

            except KeyError:
                print(f"Error in downloading {name}")

    else:
        for name in tickers.index:
            if timeframe == Mt5.TIMEFRAME_D1:
                data_dict[name] = pd.read_csv(f"Time series\\Daily\\{name}.csv", index_col=0)

            elif timeframe == Mt5.TIMEFRAME_H1:
                data_dict[name] = pd.read_csv(f"Time series\\Hourly\\{name}.csv", index_col=0)

            elif timeframe == Mt5.TIMEFRAME_M5:
                data_dict[name] = pd.read_csv(f"Time series\\Mins\\{name}.csv", index_col=0)

            else:
                raise ValueError("Timeframe not yet implemented.")

            date_time = pd.to_datetime(data_dict[name]["time"], format="%Y-%m-%d %H:%M:%S")
            data_dict[name]["time"] = date_time.map(pd.Timestamp.timestamp)
            data_dict[name].set_index("time", inplace=True)

            print(f"{name} loaded correctly")

    return data_dict


def time_series_saving(folder: str = "Daily"):
    """Save time series into a project folder."""
    global data_dict
    for name in data_dict.keys():
        data_dict[name].to_csv(f"Time series\\{folder}\\{name}.csv")
        print(f"{name} saved correctly")


def market_order(symbol: str, kind: str, act: str, lot: int = 0.1):
    """Send and execute a real trade on MetaTrader5 through the API. A response will be visible via console
    and Telegram channel."""
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


def backtest_elm():
    """ELM strategy backtest"""
    global pos_dict
    all_data = time_series_download(online=False)

    bt_return = pd.DataFrame(index=tickers.index)
    for tkr in tickers.index:
        max_dd = 0.0
        pos_dict = {}
        strategy = ELMStrategy(data=all_data[tkr],
                               ticker=tkr,
                               days=1,
                               timeframe="M",
                               save=True)
        win = 0
        returns = Portfolio(data=all_data,
                            ticker=tkr,
                            exchange=tickers.loc[tkr, "Toeuro"],
                            signals=strategy.generate_signals(),
                            stoploss=1).backtest_portfolio()
        bt_return.loc[tkr, strategy.name] = returns.iloc[-1, returns.columns.get_loc("cash")]

        for val in pos_dict.values():
            # Number of winning positions
            if val.iswinner():
                win += 1
            # Maximum drawdown
            if val.earnings() < max_dd:
                max_dd = val.earnings()
        try:
            bt_return.loc[tkr, f"{strategy.name} % Win"] = (win / len(pos_dict.keys())) * 100

        except ZeroDivisionError:
            bt_return.loc[tkr, f"{strategy.name} % Win"] = 0

        bt_return.loc[tkr, f"{strategy.name} Win"] = win
        bt_return.loc[tkr, f"{strategy.name}  Operations"] = len(pos_dict)
        bt_return.loc[tkr, "Max DD"] = max_dd
        print(f"{strategy.name} strategy on {tkr} successfully tested")

    bt_return.to_csv("ELM.csv")


class Position:
    """When a trade is done, a Position object is created. It contains all the useful
    functions and properties of the trade itself."""
    def __init__(self, ticker, exchange: str, kind: str, opens: list,
                 close=None, close_index=0.0, stoploss: float = 0.015, quantity: int = 50000):
        # open and close are lists that contain the closing prices of pair and exchange
        if isinstance(close, list):
            self.close = close
        else:
            self.close = [0.0, 0]
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
        """This function check if the position is still open or not."""
        if self.closeind == 0.0:
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

    def setclose(self, close: float, closeexc: int, closeind: float):
        """Used to close the trade."""
        self.close = [close, closeexc]
        self.closeind = closeind


class Portfolio:
    """This class creates an actual portfolio for one financial instrument and generates the positions, given a
    signal DataFrame. It's used for backtesting."""
    def __init__(self, data: dict, ticker: str, exchange: str,
                 signals: pd.DataFrame, init_capital=100000.0, stoploss: float = 1):
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
        prev_idx = 0.0
        for i in positions.index:

            if tickers.loc[self.ticker, "Base"] == "EUR":
                cross = 1.0

            else:
                try:
                    cross = self.all_data["EUR" + tickers.loc[self.ticker, "Base"]].loc[i, "close"]

                except KeyError:
                    cross = self.all_data["EUR" + tickers.loc[self.ticker, "Base"]].loc[prev_idx, "close"]

            # Determine the quantity
            if abs(self.signals.loc[i, "signal"]) == 1:
                qty = 25000

            elif abs(self.signals.loc[i, "signal"]) == 2:
                qty = 50000

            else:
                qty = 0

            # Bullish signal
            spread = self.tkr_data.loc[i, "spread"] * float(tickers.loc[self.ticker, "pip"])
            if self.signals.loc[i, "signal"] > 0:
                pos_dict[i] = Position(ticker=self.ticker,
                                       exchange=self.exchange,
                                       kind="long",
                                       opens=[self.tkr_data.loc[i, "close"] + spread, cross],
                                       stoploss=self.sl,
                                       quantity=qty)

                # Close opposite positions
                for pos in pos_dict.values():

                    if pos.getkind() == "short" and pos.isopen():
                        try:
                            exchange_close = self.all_data[self.exchange].loc[i, "close"]
                        except KeyError:
                            exchange_close = self.all_data[self.exchange].loc[prev_idx, "close"]

                        pos.setclose(self.tkr_data.loc[i, "close"] + spread,
                                     exchange_close, i)

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

                    if pos.getkind() == "long" and pos.isopen():
                        try:
                            exchange_close = self.all_data[self.exchange].loc[i, "close"]
                        except KeyError:
                            exchange_close = self.all_data[self.exchange].loc[prev_idx, "close"]

                        pos.setclose(self.tkr_data.loc[i, "close"], exchange_close, i)

            else:
                pass

            # Check stop loss
            for pos in pos_dict.values():
                if pos.isopen() \
                        and pos.getkind() == "long" \
                        and pos.sl_price() > self.tkr_data.loc[i, "close"]:
                    try:
                        exchange_close = self.all_data[self.exchange].loc[i, "close"]
                    except KeyError:
                        exchange_close = self.all_data[self.exchange].loc[prev_idx, "close"]

                    pos.setclose(self.tkr_data.loc[i, "close"],
                                 exchange_close, i)

                elif pos.isopen() \
                        and pos.getkind() == "short" \
                        and pos.sl_price() < self.tkr_data.loc[i, "close"]:
                    try:
                        exchange_close = self.all_data[self.exchange].loc[i, "close"]
                    except KeyError:
                        exchange_close = self.all_data[self.exchange].loc[prev_idx, "close"]

                    pos.setclose(self.tkr_data.loc[i, "close"] + spread,
                                 exchange_close, i)

            prev_idx = i
        # Close opened positions at the end of the backtest
        for pos in pos_dict.values():

            if pos.isopen():
                pos.setclose(self.tkr_data.loc[positions.index[-1], "close"],
                             self.all_data[self.exchange].loc[positions.index[-1], "close"],
                             positions.index[-1])

        for i in pos_dict.keys():
            positions.loc[i, "buycf"] = - pos_dict[i].buy_cashflow()
            positions.loc[pos_dict[i].getclose(), "sellcf"] = pos_dict[i].buy_cashflow() + pos_dict[i].earnings()

        positions[self.ticker] = positions["buycf"] + positions["sellcf"]

        return positions

    def backtest_portfolio(self, save: bool = True):
        portfolio = pd.DataFrame(index=self.signals.index).fillna(0.0)
        portfolio["holdings"] = self.positions[self.ticker]
        portfolio["cash"] = self.init_capital + portfolio["holdings"].cumsum()
        portfolio["total"] = portfolio["cash"] + portfolio["holdings"]
        portfolio["returns"] = portfolio["total"].pct_change() * 100
        if save:
            self.positions.to_csv(f"{ROOT_DIR}\\POSITION\\{self.ticker}positions.csv")

        return portfolio


if __name__ == "__main__":
    backtest_elm()
