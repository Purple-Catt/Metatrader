import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import stat_analysis as sa
from wingen import WindowGenerator, compile_and_fit
import data_preprocessing as dp

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
violin = False
batch = 64
epochs = 100

params = sa.parameters_hourly
data_dict = {}
tickers = pd.read_csv("Forex_ticker.csv", index_col=0)
for names in ["EURUSD"]:
    data_dict[names] = pd.read_csv(f"Time series\\Hourly\\{names}.csv", index_col=0)


MAX_EPOCHS = 100


if __name__ == "__main__":
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16, return_sequences=False),
        tf.keras.layers.Dense(units=1)])
    # Uncomment the following lines if you'd like to use the WindowGenerator class

    train, val, test, column_indices, num_features = dp.preprocessing_for_win(df=data_dict["EURUSD"])
    w1 = WindowGenerator(input_width=64, label_width=1, shift=1,
                         train_df=train, val_df=val, test_df=test,
                         label_columns=["close"])
    history = compile_and_fit(lstm_model, w1)
    '''
    # RESHAPE THE ARRAYS
    x_train, y_train, x_test, y_test = dp.preprocessing(df=data_dict["EURUSD"])
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size=batch, drop_remainder=False)
    lstm_model.compile(optimizer=tf.optimizers.Adam(),
                       loss=tf.losses.MeanSquaredError(),
                       metrics=tf.metrics.MeanAbsoluteError())
    history = lstm_model.fit(train_dataset,
                             epochs=epochs,
                             shuffle=False)
    '''
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epoch = range(1, len(loss) + 1)
    plt.plot(epoch, loss, 'y', label='Training loss')
    plt.plot(epoch, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
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
