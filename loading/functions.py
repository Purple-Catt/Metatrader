import pandas as pd
import numpy as np
import os
import MetaTrader5 as Mt5

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tickers = pd.read_csv(f"{ROOT_DIR}\\Data\\Forex_ticker.csv", index_col=0)


def time_series_download(tf, n_candles: int = 0, online: bool = True):
    data_dict = dict()

    if online:
        for name in tickers.index:
            try:
                rates = Mt5.copy_rates_from_pos(name, tf, 0, n_candles)
                rates_frame = pd.DataFrame(rates)
                rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
                data_dict[name] = rates_frame
                date_time = pd.to_datetime(data_dict[name]["time"], format="%Y-%m-%d %H:%M:%S")
                data_dict[name]["time"] = date_time.map(pd.Timestamp.timestamp)
                data_dict[name].set_index("time", inplace=True)
                print(f"{name} downloaded correctly")

            except KeyError:
                print(f"Error in downloading {name}")

    else:
        for name in tickers.index:
            if tf == Mt5.TIMEFRAME_D1:
                folder = "Daily"

            elif tf == Mt5.TIMEFRAME_H1:
                folder = "Hourly"

            elif tf == Mt5.TIMEFRAME_M5:
                folder = "Mins"

            else:
                raise NotImplementedError("Timeframe not yet implemented.")

            data_dict[name] = pd.read_csv(f"{ROOT_DIR}\\Data\\Time series\\{folder}\\{name}.csv",
                                          index_col=0)
            date_time = pd.to_datetime(data_dict[name]["time"], format="%Y-%m-%d %H:%M:%S")
            data_dict[name]["time"] = date_time.map(pd.Timestamp.timestamp)
            data_dict[name].set_index("time", inplace=True)

            print(f"{name} loaded correctly")

    return data_dict


def time_series_saving(data: dict, folder: str = "Daily"):
    """Save time series into a project folder."""
    for name in data.keys():
        data[name].to_csv(f"{ROOT_DIR}\\Data\\Time series\\{folder}\\{name}.csv")
        print(f"{name} saved correctly")


def matrix_loading(tf):
    beta_dict = dict()
    weight_dict = dict()
    for name in tickers.index:
        if tf == Mt5.TIMEFRAME_D1:
            folder = "D"

        elif tf == Mt5.TIMEFRAME_H1:
            folder = "H"

        elif tf == Mt5.TIMEFRAME_M5:
            folder = "M"

        else:
            raise NotImplementedError("Timeframe not yet implemented.")

        beta_dict[name] = np.array(pd.read_csv(
            f"{ROOT_DIR}\\Data\\Elm_matrices\\Beta\\{folder}\\{name}_Beta.csv", index_col=0))
        weight_dict[name] = np.array(pd.read_csv(
            f"{ROOT_DIR}\\Data\\Elm_matrices\\Weight\\{folder}\\{name}_Weight.csv", index_col=0))
        print(f"{name} matrices loaded correctly")

    return beta_dict, weight_dict
