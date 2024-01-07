import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz
import MetaTrader5 as Mt5
import warnings

timezone = pytz.timezone("Etc/UTC")
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)


def preprocessing_for_win(df: pd.DataFrame, violin: bool = False):
    """Preprocess data for the Window object. Ask for a DataFrame and return it divided in train, validation and
    test DataFrame, plus some other data manipulation. If violin=True, a violinplot will be shown.
    Look at the source for further information."""
    # Remove useless attributes
    df.drop(columns=["real_volume", "spread", "tick_volume"], axis=1, inplace=True)
    # Convert Timestamp into sinusoidal attributes
    date_time = pd.to_datetime(df.pop("time"), format="%Y-%m-%d %H:%M:%S")
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24*60*60
    year = 365.25 * day
    # Scaling DataFrame
    df = df.diff()
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    df.dropna(inplace=True)
    # Splitting data into train, validation and test subsets
    col_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.8)]
    test_df = df[int(n * 0.8):]
    numm_features = df.shape[1]

    # Set to True if violin plot is needed
    if violin:
        train_mean = train_df.mean()
        train_std = train_df.std()
        df_std = (df - train_mean) / train_std
        df_std = df_std.melt(var_name='Column', value_name='Scaled')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Scaled', data=df_std)
        _ = ax.set_xticklabels(df.keys(), rotation=90)
        plt.show()
        plt.clf()

    return train_df, val_df, test_df, col_indices, numm_features


def preprocessing_bt(df: pd.DataFrame, train: float = 0.8, val: float = 0.0):
    """Preprocess data and return it divided in train, validation and test arrays. If val=0.0, no validation array will
    be returned. Look at the source for further information."""
    # Remove useless attributes
    df.drop(columns=["real_volume", "spread", "tick_volume"], axis=1, inplace=True)
    # Convert Timestamp into sinusoidal attributes
    date_time = pd.to_datetime(df.pop("time"), format="%Y-%m-%d %H:%M:%S")
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24*60*60
    year = 365.25 * day
    df = df.pct_change()
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    # Splitting data into train, validation and test subsets
    y = df["close"].shift(-1)
    df = df.join(y, lsuffix='_caller', rsuffix='_other')
    df.dropna(inplace=True)
    df.rename(mapper={"close_caller": "close", "close_other": "y"}, axis="columns", inplace=True)
    n = len(df)
    if val == 0.0:
        x_train = df[:int(n * train)]
        y_train = x_train.pop("y")
        x_test = df[int(n * train):]
        y_test = x_test.pop("y")

        return (x_train.to_numpy(), y_train.to_numpy(),
                x_test.to_numpy(), y_test.to_numpy())
    else:
        x_train = df[:int(n * (train - val))]
        y_train = x_train.pop("y")
        x_val = df[int(n * (train - val)):int(n * train)]
        y_val = x_val.pop("y")
        x_test = df[int(n * train):]
        y_test = x_test.pop("y")

        return (x_train.to_numpy(), y_train.to_numpy(),
                x_val.to_numpy(), y_val.to_numpy(),
                x_test.to_numpy(), y_test.to_numpy())


def preprocessing(df: pd.DataFrame):
    """Preprocess data for live trading use. Look at the source for further information."""
    # Remove useless attributes
    df.drop(columns=["real_volume", "spread", "tick_volume"], axis=1, inplace=True)
    # Convert Timestamp into sinusoidal attributes
    date_time = pd.to_datetime(df.pop("time"), format="%Y-%m-%d %H:%M:%S")
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.25 * day
    df = df.diff()

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


def elm_preprocessing(df: pd.DataFrame, days: int = 5, timeframe: str = "M", live: bool = False):
    """Preprocess data for live trading use. Look at the source for further information.\n
    Parameters:\n
    df: The raw dataframe to process\n
    days: The number of past days to consider\n
    timeframe: Timeframe of the series used, string between D, H, M, respectively for daily, hourly and 5 minutes data\n
    live: Boolean value; if True, the last row - aka the actual price - is returned separetly to make predictions."""
    if timeframe == "D":
        depth = days - 1
    elif timeframe == "H":
        depth = (days * 24) - 1
    elif timeframe == "M":
        depth = (days * 12 * 24) - 1
    else:
        raise ValueError(f"String between 'D', 'H' or 'M' expected, got {timeframe} instead.")

    # Remove useless attributes
    df.drop(columns=["real_volume", "spread", "tick_volume"], axis=1, inplace=True)
    past = [df]
    for i in range(depth):
        past.append(df.shift(i + 1).rename(
            columns={"open": f"{i}open", "high": f"{i}high", "low": f"{i}low", "close": f"{i}close"})
        )

    res = pd.concat(past, axis=1)
    res["y"] = res["close"].shift(-1)
    if live:
        res, last = res.drop(res.tail(1).index), res.tail(1)
        last.drop("y", axis=1, inplace=True)

    res.dropna(inplace=True)
    y = res.pop("y")
    if live:
        return res, np.array(y), np.array(last)

    else:
        return res, np.array(y)


def from_tick_preprocessing(tickers, data_dict: dict):
    """
    The 'data_cleaning' function download 24h ticks data from MT5 and clean them as showed in
    'Realized Kernels in Practice: Trades and Quotes Barndorff-Nielsen et al. 2008b'.\n
    It must be used for the RV strategy, that still needs to be implemented in this code.
    """
    sec_data = {}
    ret_data = {}
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

        return data_dict, sec_data, ret_data
