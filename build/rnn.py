from keras.layers import Dense, LSTM
from keras.models import Sequential
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from build.wingen import WindowGenerator, compile_and_fit
import data_preprocessing as dp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


def create_model(window):
    model = Sequential([
        LSTM(32, return_sequences=True),
        LSTM(16, return_sequences=False, dropout=0.1),
        Dense(units=1)])

    history = compile_and_fit(model, window)

    return history, model


def plot_learning_curve(history, start_epoch=1):
    lgd = ['Loss TR']
    epochs = len(history["loss"])
    plt.plot(range(start_epoch, epochs), history['loss'][start_epoch:])

    if "val_loss" in history:
        plt.plot(range(start_epoch, epochs), history['val_loss'][start_epoch:])
        lgd.append('Loss VL')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f'Keras Learning Curve')
    plt.legend(lgd)

    # Check if predictions are available in the history
    if 'predictions' in history:
        predictions = history['predictions']

        # Plot predictions for each variable
        for i in range(predictions.shape[1]):
            plt.figure()
            plt.plot(range(start_epoch, epochs), predictions[:, i][start_epoch:])
            plt.xlabel("Epoch")
            plt.ylabel(f"Prediction Variable {i + 1}")
            plt.title(f'Keras Learning Curve \n - Prediction Variable {i + 1}')

    plt.show()


def training(plot: bool = True, save: bool = True):
    data_dict = {}
    tickers = pd.read_csv(f"{ROOT_DIR}\\Data\\Forex_ticker.csv", index_col=0)

    for names in tickers.index:
        data_dict[names] = pd.read_csv(f"{ROOT_DIR}\\Data\\Time series\\Hourly\\{names}.csv", index_col=0)

    for key in tickers.index:
        train, val, test, column_indices, num_features = dp.preprocessing_for_win(df=data_dict[key])
        w1 = WindowGenerator(input_width=64, label_width=1, shift=1,
                             train_df=train, val_df=val, test_df=test,
                             label_columns=["close"])
        res, lstm_model = create_model(w1)

        if plot:
            plot_learning_curve(history=res.history)

        scores = lstm_model.evaluate(w1.test, verbose=0)
        print("%s model: %s: %.2f%%" % (key, lstm_model.metrics_names[1], scores[1] * 100))
        if save:
            lstm_model.save(f"{ROOT_DIR}\\Data\\Models\\{key}.keras")
            print(f"Saved {key} model to disk")


def prediction(model, x):
    pred = model.predict(x=x)

    return float(pred)


if __name__ == "__main__":
    training()
