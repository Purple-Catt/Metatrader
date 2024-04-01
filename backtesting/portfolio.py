import os
import pandas as pd
from backtesting.position import Position

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tickers = pd.read_csv(f"{ROOT_DIR}\\Data\\Forex_ticker.csv", index_col=0)


class Portfolio:
    """This class creates an actual portfolio for one financial instrument and generates the positions, given a
    signal DataFrame. It's used for backtesting."""
    def __init__(self, data: dict, ticker: str, exchange: str,
                 signals: pd.DataFrame, positions: dict, init_capital=100000.0, stoploss: float = 1):
        self.tkr_data = data[ticker]
        self.all_data = data
        self.ticker = ticker
        self.exchange = exchange
        self.signals = signals
        self.pos_dict = positions
        self.init_capital = init_capital
        self.sl = stoploss
        self.positions = self.generate_positions()

    def generate_positions(self):
        """Open long (short) positions based on signals and eventually close already opened ones."""
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.ticker] = 0.0
        positions["buycf"] = 0.0
        positions["sellcf"] = 0.0
        prev_idxs = list()
        for i in positions.index:
            prev_idxs.append(i)

            if tickers.loc[self.ticker, "Base"] == "EUR":
                cross = 1.0

            else:
                for j in reversed(prev_idxs):
                    try:
                        cross = self.all_data["EUR" + tickers.loc[self.ticker, "Base"]].loc[j, "close"]
                        break

                    except KeyError:
                        pass

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
                self.pos_dict[i] = Position(ticker=self.ticker,
                                            exchange=self.exchange,
                                            kind="long",
                                            opens=[self.tkr_data.loc[i, "close"] + spread, cross],
                                            stoploss=self.sl,
                                            quantity=qty)

                # Close opposite positions
                for pos in self.pos_dict.values():

                    if pos.getkind() == "short" and pos.isopen():
                        for j in reversed(prev_idxs):
                            try:
                                exchange_close = self.all_data[self.exchange].loc[j, "close"]
                                break

                            except KeyError:
                                pass

                        pos.setclose(self.tkr_data.loc[i, "close"] + spread,
                                     exchange_close, i)

            # Bearish signal
            elif self.signals.loc[i, "signal"] < 0:
                self.pos_dict[i] = Position(ticker=self.ticker,
                                            exchange=self.exchange,
                                            kind="short",
                                            opens=[self.tkr_data.loc[i, "close"], cross],
                                            stoploss=self.sl,
                                            quantity=qty)

                # Close opposite positions
                for pos in self.pos_dict.values():

                    if pos.getkind() == "long" and pos.isopen():
                        for j in reversed(prev_idxs):
                            try:
                                exchange_close = self.all_data[self.exchange].loc[j, "close"]
                                break

                            except KeyError:
                                pass

                        pos.setclose(self.tkr_data.loc[i, "close"], exchange_close, i)

            else:
                pass

            # Check stop loss
            for pos in self.pos_dict.values():
                if pos.isopen() \
                        and pos.getkind() == "long" \
                        and pos.sl_price() > self.tkr_data.loc[i, "close"]:
                    for j in reversed(prev_idxs):
                        try:
                            exchange_close = self.all_data[self.exchange].loc[j, "close"]
                            break

                        except KeyError:
                            pass

                    pos.setclose(self.tkr_data.loc[i, "close"],
                                 exchange_close, i)

                elif pos.isopen() \
                        and pos.getkind() == "short" \
                        and pos.sl_price() < self.tkr_data.loc[i, "close"]:
                    for j in reversed(prev_idxs):
                        try:
                            exchange_close = self.all_data[self.exchange].loc[j, "close"]
                            break

                        except KeyError:
                            pass

                    pos.setclose(self.tkr_data.loc[i, "close"] + spread,
                                 exchange_close, i)

        # Close opened positions at the end of the backtest
        for pos in self.pos_dict.values():

            if pos.isopen():
                pos.setclose(self.tkr_data.loc[positions.index[-1], "close"],
                             self.all_data[self.exchange].loc[positions.index[-1], "close"],
                             positions.index[-1])

        for i in self.pos_dict.keys():
            positions.loc[i, "buycf"] = - self.pos_dict[i].buy_cashflow()
            positions.loc[self.pos_dict[i].getclose(), "sellcf"] = (self.pos_dict[i].buy_cashflow() +
                                                                    self.pos_dict[i].earnings())

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
