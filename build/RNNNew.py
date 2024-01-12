import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN
from keras.regularizers import l1
import keras.layers
from build.alphaRNN import *
from build.alphatRNN import *
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save = False
load = False
cross_val = False  # Warning: Changing this to True will take several hours to run
use_features = ['close']  # continuous input
use_feature = 'close'
target = 'close'  # continuous output
df = pd.read_csv(f"{ROOT_DIR}\\Data\\Time series\\Hourly\\EURUSD.csv", index_col=0)


def get_lagged_features(value, n_steps, n_steps_ahead):
    """
    value: feature value to be lagged
    n_steps: number of lags, i.e. sequence length
    n_steps_ahead: forecasting horizon
    """
    lag_list = []
    for lag in range(n_steps+n_steps_ahead-1, n_steps_ahead-1, -1):
        lag_list.append(value.shift(lag))
    return pd.concat(lag_list, axis=1)


def AlphatRNN_(n_units=10, l1_reg=0, seed=0):
    model = Sequential()
    model.add(AlphatRNN(n_units, activation='tanh',
                        recurrent_activation='sigmoid',
                        kernel_initializer=keras.initializers.glorot_uniform(seed),
                        bias_initializer=keras.initializers.glorot_uniform(seed),
                        recurrent_initializer=keras.initializers.orthogonal(seed),
                        kernel_regularizer=l1(l1_reg),
                        input_shape=(x_train_reg.shape[1], x_train_reg.shape[-1]),
                        unroll=True))
    model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed),
                    bias_initializer=keras.initializers.glorot_uniform(seed),
                    kernel_regularizer=l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def AlphaRNN_(n_units=10, l1_reg=0, seed=0):
    model = Sequential()
    model.add(AlphaRNN(n_units, activation='tanh',
                       kernel_initializer=keras.initializers.glorot_uniform(seed),
                       bias_initializer=keras.initializers.glorot_uniform(seed),
                       recurrent_initializer=keras.initializers.orthogonal(seed),
                       kernel_regularizer=l1(l1_reg),
                       input_shape=(x_train_reg.shape[1], x_train_reg.shape[-1]),
                       unroll=True))
    model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed),
                    bias_initializer=keras.initializers.glorot_uniform(seed),
                    kernel_regularizer=l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def SimpleRNN_(n_units=10, l1_reg=0, seed=0):
    model = Sequential()
    model.add(SimpleRNN(n_units, activation='tanh',
                        kernel_initializer=keras.initializers.glorot_uniform(seed),
                        bias_initializer=keras.initializers.glorot_uniform(seed),
                        recurrent_initializer=keras.initializers.orthogonal(seed),
                        kernel_regularizer=l1(l1_reg),
                        input_shape=(x_train_reg.shape[1], x_train_reg.shape[-1]),
                        unroll=True,
                        stateful=False))
    model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed),
                    bias_initializer=keras.initializers.glorot_uniform(seed),
                    kernel_regularizer=l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def GRU_(n_units=10, l1_reg=0, seed=0):
    model = Sequential()
    model.add(GRU(n_units, activation='tanh',
                  kernel_initializer=keras.initializers.glorot_uniform(seed),
                  bias_initializer=keras.initializers.glorot_uniform(seed),
                  recurrent_initializer=keras.initializers.orthogonal(seed),
                  kernel_regularizer=l1(l1_reg),
                  input_shape=(x_train_reg.shape[1], x_train_reg.shape[-1]),
                  unroll=True))
    model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed),
                    bias_initializer=keras.initializers.glorot_uniform(seed),
                    kernel_regularizer=l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def LSTM_(n_units=10, l1_reg=0, seed=0):
    model = Sequential()
    model.add(LSTM(n_units, activation='tanh',
                   kernel_initializer=keras.initializers.glorot_uniform(seed),
                   bias_initializer=keras.initializers.glorot_uniform(seed),
                   recurrent_initializer=keras.initializers.orthogonal(seed),
                   kernel_regularizer=l1(l1_reg),
                   input_shape=(x_train_reg.shape[1], x_train_reg.shape[-1]),
                   unroll=True))
    model.add(Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed),
                    bias_initializer=keras.initializers.glorot_uniform(seed),
                    kernel_regularizer=l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


pacf = sm.tsa.stattools.pacf(df[use_features], nlags=30)
n_steps = np.where(np.array(np.abs(pacf) > 2.58/np.sqrt(len(df[use_features]))) is False)[0][0] - 1
print(n_steps)
plt.plot(pacf, label='pacf')
plt.plot([2.58/np.sqrt(len(df[use_features]))]*30, label='99% confidence interval (upper)')
plt.plot([-2.58/np.sqrt(len(df[use_features]))]*30, label='99% confidence interval (lower)')
plt.legend()
plt.show()
input()
train_weight = 0.8
split = int(len(df) * train_weight)
df_train = df.iloc[:split]
mu = float(df_train[use_features].mean())
sigma = float(df_train[use_features].std())
df_train = df_train[use_features].apply(lambda x: (x - mu) / sigma)
df_test = df.iloc[split:]
df_test = df[use_features].apply(lambda x: (x - mu) / sigma).iloc[split:]

n_steps_ahead = 10

x_train_list = []
for use_feature in use_features:
    x_train_reg = get_lagged_features(df_train, n_steps, n_steps_ahead).dropna()
    x_train_list.append(x_train_reg)
x_train_reg = pd.concat(x_train_list, axis=1)

col_ords = []
for i in range(n_steps):
    for j in range(len(use_features)):
        col_ords.append(i + j * n_steps)

x_train_reg = x_train_reg.iloc[:, col_ords]
y_train_reg = df_train.loc[x_train_reg.index, [target]].values
x_train_reg = np.reshape(x_train_reg.values, (x_train_reg.shape[0],
                                              int(x_train_reg.shape[1] / len(use_features)), len(use_features)))
y_train_reg = np.reshape(y_train_reg, (y_train_reg.shape[0], 1, 1))

x_test_list = []
for use_feature in use_features:
    x_test_reg = get_lagged_features(df_test, n_steps, n_steps_ahead).dropna()
    x_test_list.append(x_test_reg)
x_test_reg = pd.concat(x_test_list, axis=1)

x_test_reg = x_test_reg.iloc[:, col_ords]
y_test_reg = df_test.loc[x_test_reg.index, [target]].values
x_test_reg = np.reshape(x_test_reg.values, (x_test_reg.shape[0],
                                            int(x_test_reg.shape[1]/len(use_features)), len(use_features)))

y_test_reg = np.reshape(y_test_reg, (y_test_reg.shape[0], 1, 1))

train_batch_size = y_train_reg.shape[0]
test_batch_size = y_test_reg.shape[0]
time_size = y_train_reg.shape[1]

x_train_reg = pd.concat(x_train_list, axis=1)
x_train_reg = x_train_reg.iloc[:, col_ords]
y_train_reg = df_train.loc[x_train_reg.index, [target]].values
x_train_reg = np.reshape(x_train_reg.values, (x_train_reg.shape[0],
                                              int(x_train_reg.shape[1] / len(use_features)), len(use_features)))

x_train_reg = pd.concat(x_train_list, axis=1)
x_train_reg = x_train_reg.iloc[:, col_ords]
y_train_reg = df_train.loc[x_train_reg.index, [target]].values
x_train_reg = np.reshape(x_train_reg.values, (x_train_reg.shape[0],
                                              int(x_train_reg.shape[1] / len(use_features)), len(use_features)))

max_epoches = 2000
batch_size = 1000

es = tf.keras.callback.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=100, min_delta=1e-7, restore_best_weights=True)

params = {'rnn': {'model': '', 'function': SimpleRNN_, 'l1_reg': 0.0, 'H': 5, 'color': 'blue', 'label': 'RNN'},
          'alpharnn': {'model': '', 'function': AlphaRNN_, 'l1_reg': 0.0, 'H': 10, 'color': 'green',
                       'label': '$\\alpha$-RNN'},
          'alphatrnn': {'model': '', 'function': AlphatRNN_, 'l1_reg': 0.0, 'H': 5, 'color': 'cyan',
                        'label': '$\\alpha_t$-RNN'},
          'gru': {'model': '', 'function': GRU_, 'l1_reg': 0.0, 'H': 20, 'color': 'orange', 'label': 'GRU'},
          'lstm': {'model': '', 'function': LSTM_, 'l1_reg': 0.0, 'H': 10, 'color': 'red', 'label': 'LSTM'}
          }

if cross_val:
    n_units = [5, 10, 20]
    l1_reg = [0, 0.001, 0.01, 0.1]
    tscv = TimeSeriesSplit(n_splits = 5)
    param_grid = dict(n_units=n_units,l1_reg=l1_reg)

    for key in params.keys():  # params[key]['function']
        model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=params[key]['function'], epochs=max_epoches, batch_size=batch_size, verbose=2)
        grid = GridSearchCV(estimator=model,param_grid=param_grid, cv=tscv, n_jobs=1, verbose=2)
        grid_result = grid.fit(x_train_reg[:30000],y_train_reg[:30000],callbacks=[es])
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params_ = grid_result.cv_results_['params']
        for mean, stdev, param_ in zip(means, stds, params_):
            print("%f (%f) with %r" % (mean, stdev, param_))

        params[key]['H'] = grid_result.best_params_['n_units']
        params[key]['l1_reg']= grid_result.best_params_['l1_reg']

for key in params.keys():
    tf.set_random_seed(0)
    model = params[key]['function'](params[key]['H'], params[key]['l1_reg'])
    model.fit(x_train_reg, y_train_reg, epochs=max_epoches, batch_size=batch_size, callbacks=[es], shuffle=False)
    params[key]['model'] = model

# optionally save the fitted model
if save is True:
    for key in params.keys():
        params[key]['model'].save(key + '.hdf5', overwrite=True)  # creates a HDF5 file

if load is True:
    # optionally load the fitted model
    for key in params.keys():
        params[key]['model'] = tf.keras.models.load_model(key + '.hdf5')

# Print out the value of alpha \in [0,1] for the alpha-RNN model
model = params['alpharnn']['model']
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

for name, weight in zip(names, weights):
    if name == 'alpha_rnn_32/alpha:0':
        print("alpha= " + str(sigmoid(weight)))

upper = 5000
for key in params.keys():
    model = params[key]['model']
    model.summary()

    params[key]['MSE_train'] = mean_squared_error(df_train[use_feature][n_steps+n_steps_ahead-1:],
                                                  model.predict(x_train_reg, verbose=1))
    params[key]['predict'] = model.predict(x_test_reg, verbose=1)
    params[key]['MSE_test'] = mean_squared_error(df_test[use_feature][n_steps+n_steps_ahead-1:upper],
                                                 params[key]['predict'][:upper-(n_steps+n_steps_ahead-1)])

fig = plt.figure(figsize=(12,7))
plt.plot(df_test.index[n_steps+n_steps_ahead-1:upper], df_test[use_feature][n_steps+n_steps_ahead-1:upper], color="black", label="Observed")

for key in params.keys():
    plt.plot(df_test.index[n_steps+n_steps_ahead-1:upper], params[key]['predict'][:upper-(n_steps+n_steps_ahead-1), 0], color=params[key]['color'], label=params[key]['label'])
plt.legend(loc="best", fontsize=12)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Y', fontsize=20)
plt.title('Observed vs Model (Testing)', fontsize=16)
plt.savefig("Obs_vs_model.png")
plt.clf()

fig = plt.figure(figsize=(12,7))

for key in params.keys():
    plt.plot(df_test.index[n_steps+n_steps_ahead-1:upper], df_test[use_feature][n_steps+n_steps_ahead-1:upper]-params[key]['predict'][:upper-(n_steps+n_steps_ahead-1), 0], color=params[key]['color'], label=params[key]['label'] + " (" +  str(round(params[key]['MSE_test'],7)) +")")

plt.legend(loc="best", fontsize=12)
plt.title('Observed vs Model Error (Training)', fontsize=16)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Y-$\\hat{Y}$', fontsize=20)
plt.savefig("Obs_vs_model_err.png")
