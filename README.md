# Trading-framework
The aim of this project is to provide a framework to backtest and trade **in live** different strategies on Forex market using Metatrader5 as trading platform.
The purposes of the different files are the following:
- **Time series folder** → It provides time series in different timeframes (daily, minutes, ticks) useful for backtesting purposes. The "Cleaned" folder contains time series useful for RV strategies (see main.py for further details).
- **Forex_ticker** → It contains the tickers of the crosses exchanged and other useful info (broker leverage, pip, currencies etc.).
- **Parameter_daily/hourly** → They store the _p,d,q_ and distribution parameters calculated for the ARIMA-GARCH strategy into a _.csv_ file.

### Python files
- **main** → Main script that allow to backtest and live testing trading strategies.
- **stat_analysis** → It provides several functions useful for strategies based on econometrics and/or statistical inference. The RV functions are actually incomplete.
- **Indicators** → It provides several functions useful for strategies based on technical analysis.
- **data_preprocessing** → It provides functions for the preprocessing and manipulation of the datasets.
- **wingen** → Contains the _WindowGenerator_ class retrieved from the TensorFlow website.
- **RNN** → Implement strategies using Neural Network. Actually it includes a RNN.
- **telegram_bot** → Allow to handle a Telegram bot to send messages in a group.
- **Testing** → It exists only for development purposes.
