from datetime import datetime
import os
import MetaTrader5 as Mt5
import warnings
from keras.models import load_model
from build import stat_analysis as sa
from loading.functions import time_series_download, time_series_saving, matrix_loading
from backtesting.portfolio import Portfolio
import strategies
from strategies import ArmaGarchStrategy, RNNStrategy, ELMStrategy
import pytz
import telegram_bot as tb
import credentials
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Broker credentials
real = credentials.real
demo = credentials.demo
# Variables
sign_dict = {}  # Signals for backtestings
models = {}  # Models for NN strategies
last_sign = strategies.last_sign
tickers = pd.read_csv(f"{ROOT_DIR}\\Data\\Forex_ticker.csv", index_col=0)
timezone = pytz.timezone("Etc/UTC")
timeframe = Mt5.TIMEFRAME_H1

# Arima-Garch model parameters
params = sa.parameters
# params = sa.parameters_hourly

candles = 1000  # Number of OHLCV data to download


def metatrader_start(username, psw, svr):
    """Establish connection to the MetaTrader 5 terminal."""
    if not Mt5.initialize():
        print("initialize() failed, error code =", Mt5.last_error())
        quit()
    authorized = Mt5.login(username, password=psw, server=svr)
    if authorized:
        print("Connection established")
    else:
        print(f"failed to connect at account #{username}, error code: {Mt5.last_error()}")


def market_order(symbol: str, kind: str, act: str, lot: int = 0.05):
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
            tb.send_message(f"Position on _{symbol}_ closed succesfully.\n*Profit*: {profit}")


def backtest_elm():
    """ELM strategy backtest."""
    all_data = time_series_download(tf=timeframe, online=False)
    # betas, weights = matrix_loading()

    bt_return = pd.DataFrame(index=tickers.index)
    for ticker in tickers.index:
        max_dd = 0.0
        pos_dict = {}
        strat = ELMStrategy(data=all_data[ticker],
                            ticker=ticker,
                            weight="xavier",
                            days=1,
                            timeframe="H",
                            depth=2,
                            save=False,
                            returns=True,
                            scale=False)
        win = 0
        returns = Portfolio(data=all_data,
                            ticker=ticker,
                            exchange=tickers.loc[ticker, "Toeuro"],
                            signals=strat.generate_signals(),
                            positions=pos_dict,
                            stoploss=1).backtest_portfolio()
        bt_return.loc[ticker, strat.name] = returns.iloc[-1, returns.columns.get_loc("cash")]

        for val in pos_dict.values():
            # Number of winning positions
            if val.iswinner():
                win += 1
            # Maximum drawdown
            if val.earnings() < max_dd:
                max_dd = val.earnings()
        try:
            bt_return.loc[ticker, f"{strat.name} % Win"] = (win / len(pos_dict.keys())) * 100

        except ZeroDivisionError:
            bt_return.loc[ticker, f"{strat.name} % Win"] = 0

        bt_return.loc[ticker, f"{strat.name} Win"] = win
        bt_return.loc[ticker, f"{strat.name}  Operations"] = len(pos_dict)
        bt_return.loc[ticker, "Max DD"] = max_dd
        print(f"{strat.name} strategy on {ticker} successfully tested")

    bt_return.to_csv("ELM.csv")


def backtest_armagarch(saveret: bool = False):
    """Arima-Garch strategy backtest"""
    metatrader_start(demo[0], demo[1], demo[2])
    all_data = time_series_download(n_candles=candles, online=True, tf=timeframe)
    Mt5.shutdown()
    bt_optimization = pd.DataFrame(index=tickers.index)
    for sl in [100]:
        bt_return = pd.DataFrame(index=tickers.index)
        for ticker in tickers.index:
            max_dd = 0.0
            if params.loc[ticker, "distribution"] == "0":
                pass
            else:
                pos_dict = {}
                models[ticker] = load_model(f"{ROOT_DIR}\\Data\\Models\\{ticker}.keras")
                strat = ArmaGarchStrategy(all_data[ticker], ticker=ticker, parameters=params)
                win = 0
                returns = Portfolio(data=all_data,
                                    ticker=ticker,
                                    exchange=tickers.loc[ticker, "Toeuro"],
                                    signals=strat.generate_signals(),
                                    positions=pos_dict,
                                    stoploss=sl).backtest_portfolio()
                if saveret:
                    returns.to_csv(f"Portfolios\\{strat.name}\\{ticker}.csv")

                bt_return.loc[ticker, strat.name] = returns.loc[candles - 1, "cash"]
                for val in pos_dict.values():
                    # Number of winning positions
                    if val.iswinner() is True:
                        win += 1
                    # Maximum drawdown
                    if val.earnings() < max_dd:
                        max_dd = val.earnings()
                bt_return.loc[ticker, f"{strat.name} % Win"] = (win / len(pos_dict.keys())) * 100
                bt_return.loc[ticker, f"{strat.name} Win"] = win
                bt_return.loc[ticker, f"{strat.name}  Operations"] = len(pos_dict)
                bt_return.loc[ticker, "Max DD"] = max_dd
                print(f"{sl} {strat.name} strategy on {ticker} successfully tested")
                bt_optimization.loc[ticker, f"Ret {sl}"] = returns.loc[candles - 1, "cash"] - 100000
                bt_optimization.loc[ticker, f"MaxDD {sl}"] = max_dd
    bt_optimization.to_csv("Optimization.csv")


def backtest_rnn(loadmodel: bool = False):
    """RNN strategy backtest"""
    all_data = time_series_download(online=False, tf=timeframe)

    bt_return = pd.DataFrame(index=tickers.index)
    for ticker in tickers.index:
        max_dd = 0.0
        pos_dict = {}
        if loadmodel:
            models[ticker] = load_model(f"{ROOT_DIR}\\Data\\Models\\{ticker}.keras")

        strat = RNNStrategy(data=all_data[ticker], ticker=ticker, model=models[ticker])
        win = 0
        returns = Portfolio(data=all_data,
                            ticker=ticker,
                            exchange=tickers.loc[ticker, "Toeuro"],
                            signals=strat.generate_signals(),
                            stoploss=1,
                            positions=pos_dict).backtest_portfolio()

        bt_return.loc[ticker, strat.name] = returns.iloc[-1, returns.columns.get_loc("cash")]

        for val in pos_dict.values():
            # Number of winning positions
            if val.iswinner():
                win += 1
            # Maximum drawdown
            if val.earnings() < max_dd:
                max_dd = val.earnings()
        try:
            bt_return.loc[ticker, f"{strat.name} % Win"] = (win / len(pos_dict.keys())) * 100

        except ZeroDivisionError:
            bt_return.loc[ticker, f"{strat.name} % Win"] = 0

        bt_return.loc[ticker, f"{strat.name} Win"] = win
        bt_return.loc[ticker, f"{strat.name}  Operations"] = len(pos_dict)
        bt_return.loc[ticker, "Max DD"] = max_dd
        print(f"{strat.name} strategy on {ticker} successfully tested")

    bt_return.to_csv("RNN.csv")


def live_trading(tf=timeframe):
    betas, weights = matrix_loading(tf=timeframe)
    while True:
        dt_now = datetime.now(timezone)
        if tf == Mt5.TIMEFRAME_D1:
            now = int(dt_now.strftime("%H%M%S"))
            runtime = [221500]

        elif tf == Mt5.TIMEFRAME_H1:
            now = int(dt_now.strftime("%M%S"))
            runtime = [5915]

        elif tf == Mt5.TIMEFRAME_M5:
            now = int(dt_now.strftime("%M%S"))
            runtime = list(range(415, 6015, 500))

        else:
            raise NotImplementedError("Timeframe not yet implemented.")

        if now in runtime:
            metatrader_start(demo[0], demo[1], demo[2])
            all_data = time_series_download(n_candles=candles, tf=timeframe)

            for ticker in tickers.index:
                try:
                    last_sign[ticker] = Mt5.positions_get(symbol=ticker)[0].type
                except AttributeError:
                    last_sign[ticker] = -1
                except IndexError:
                    last_sign[ticker] = -1
                # Uncomment the strategy you want to use
                '''
                strategy = ArmaGarchStrategy(all_data[tkr],
                                             ticker=tkr,
                                             parameters=params)

                strategy = RNNStrategy(data=all_data[tkr],
                                       ticker=tkr,
                                       model=models[tkr])
                '''
                strategy = ELMStrategy(all_data[ticker],
                                       ticker=ticker,
                                       beta=betas[ticker],
                                       weight=weights[ticker],
                                       days=5,
                                       timeframe="H",
                                       live=True,
                                       save=False)
                act_signal = strategy.live_signals(strategy.last)
                # Open long position
                if act_signal > 0:
                    opened = Mt5.positions_get(symbol=ticker)

                    if opened is not None:
                        # Close already opened position
                        for _ in opened:
                            market_order(symbol=ticker, kind="short", act="close")

                    market_order(symbol=ticker, kind="long", act="open")

                # Open short position
                elif act_signal < 0:
                    opened = Mt5.positions_get(symbol=ticker)

                    if opened is not None:
                        # Close already opened position
                        for _ in opened:
                            market_order(symbol=ticker, kind="long", act="close")

                        market_order(symbol=ticker, kind="short", act="open")

            Mt5.shutdown()


if __name__ == "__main__":
    backtest_rnn()
