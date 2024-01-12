# Trading-framework
The aim of this project is to provide a framework to backtest and trade **in live** different strategies on Forex market using Metatrader5 as trading platform.
The purposes of the different files are the following:
- **Data folder** → Time series in different timeframes, NN models and several parameters.
- **backtesting** → Portfolio and Position classes used for backtesting.
- **build** → Miscellaneous Python script used to develop strategies.
- **loading** → Functions to load data into scripts.

### Python files
- **main** → Main script that allow to backtest and live testing trading strategies.
- **strategies** → It contains all the strategy classes used in the _main_ script.
- **data_preprocessing** → It provides functions for the preprocessing and manipulation of the datasets.
- **telegram_bot** → Allow to handle a Telegram bot to send messages in a group.
