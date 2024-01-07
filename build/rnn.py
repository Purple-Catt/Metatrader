import keras
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.regularizers import L2
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from build.wingen import WindowGenerator, compile_and_fit
import data_preprocessing as dp

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


def create_model(window):
    model = Sequential([
        LSTM(16, return_sequences=True),
        LSTM(8, return_sequences=False, dropout=0.1),
        Dense(units=1, kernel_regularizer=L2())])

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


def training(plot: bool = True):
    data_dict = {}
    tickers = pd.read_csv("../Forex_ticker.csv", index_col=0)

    for names in tickers.index:
        data_dict[names] = pd.read_csv(f"Time series\\Hourly\\{names}.csv", index_col=0)

    for key in ["USDNOK"]:
        # Uncomment the following lines if you'd like to use the WindowGenerator class
        train, val, test, column_indices, num_features = dp.preprocessing_for_win(df=data_dict[key])
        w1 = WindowGenerator(input_width=64, label_width=1, shift=1,
                             train_df=train, val_df=val, test_df=test,
                             label_columns=["close"])
        res, lstm_model = create_model(w1)

        if plot:
            plot_learning_curve(history=res.history)

        scores = lstm_model.evaluate(w1.test, verbose=0)
        print("%s model: %s: %.2f%%" % (key, lstm_model.metrics_names[1], scores[1] * 100))
        lstm_model.save(f"Models\\{key}.keras")
        print(f"Saved {key} model to disk")


def prediction(model, x):
    pred = model.predict(x=x)

    return float(pred)


if __name__ == "__main__":
    training()
    '''
    # RESHAPE THE ARRAYS
    x_train, y_train, x_test, y_test = dp.preprocessing(df=data_dict[key])
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size=batch, drop_remainder=False)
    lstm_model.compile(optimizer=tf.optimizers.Adam(),
                       loss=tf.losses.MeanSquaredError(),
                       metrics=tf.metrics.MeanAbsoluteError())
    history = lstm_model.fit(train_dataset,
                             epochs=epochs,
                             shuffle=False)
    '''
    # PLOT PERFORMANCE
    '''
    val_performance = {}
    performance = {}
    val_performance['LSTM1'] = lstm_model1.evaluate(w1.val)
    performance['LSTM1'] = lstm_model1.evaluate(w1.test, verbose=0)

    x = np.arange(len(performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    metric_index = lstm_model1.metrics_names.index('mean_absolute_error')
    val_mae = [v[metric_index] for v in val_performance.values()]
    test_mae = [v[metric_index] for v in performance.values()]

    plt.ylabel('mean_absolute_error [Closing price, normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(),
               rotation=45)
    _ = plt.legend()
    plt.show()
    '''
