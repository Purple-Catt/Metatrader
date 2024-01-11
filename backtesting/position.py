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
