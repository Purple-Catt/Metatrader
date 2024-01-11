import numpy as np
from numpy.random import standard_normal, normal, uniform
import jax.numpy as jnp
import time


def _mean_squared_error(y, pred):
    return 0.5 * np.mean((y - pred) ** 2)


def _mean_abs_error(y, pred):
    return np.mean(np.abs(y, pred))


def _euclidean_error(y_true, y_pred):
    return np.sum(np.sqrt(np.sum(np.square(y_true - y_pred), axis=1))) / len(y_true)


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _tanh(x):
    return np.tanh(x)


def _fourier(x):
    return np.sin(x)


def _hardlimit(x):
    return (x >= 0).astype(int)


def _identity(x):
    return x


def _standard_normal(size):
    return standard_normal(size)


def _xavier(size):
    bound = (np.sqrt(6) / np.sqrt(size[0] + size[1]))
    return uniform(low=-bound, high=bound, size=size)


def _he_normal(size):
    return normal(loc=0.0, scale=np.sqrt(2 / size[0]), size=size)


def get_activation(name):
    return {
        'sigmoid': _sigmoid,
        'fourier': _fourier,
        'hardlimit': _hardlimit,
        'tanh': _tanh
    }[name]


def get_loss(name):
    return {
        'mse': _mean_squared_error,
        'mae': _mean_abs_error,
        'mee': _euclidean_error
    }[name]


def get_init(name: str, size: tuple):
    return {
        'std': _standard_normal(size),
        'xavier': _xavier(size),
        'he': _he_normal(size)
    }[name]


class ELM:
    def __init__(self, num_input_nodes, num_hidden_units, num_out_units, activation='sigmoid',
                 loss='mse', beta_init=None, w_init=None, bias_init=None, verbose: bool = True):
        self._num_input_nodes = num_input_nodes
        self._num_hidden_units = num_hidden_units
        self._num_out_units = num_out_units

        self._activation = get_activation(activation)
        self._loss = get_loss(loss)

        if isinstance(beta_init, np.ndarray):
            self._beta = beta_init

            if verbose:
                print("Beta initializer loaded correctly")

        else:
            self._beta = np.random.uniform(-1., 1., size=(self._num_hidden_units, self._num_out_units))

        if isinstance(w_init, str):
            try:
                self._w = get_init(w_init, size=(self._num_input_nodes, self._num_hidden_units))

            except KeyError:
                print(f"String between 'std', 'xavier' or 'he' expected, got {w_init} instead.")

        elif isinstance(w_init, np.ndarray):
            self._w = w_init

            if verbose:
                print("Weight matrix loaded correctly")
        else:
            self._w = np.random.uniform(-1, 1, size=(self._num_input_nodes, self._num_hidden_units))

        if isinstance(bias_init, np.ndarray):
            self._bias = bias_init
        else:
            self._bias = np.zeros(shape=(self._num_hidden_units,))

        if verbose:
            print('Bias shape:', self._bias.shape)
            print('W shape:', self._w.shape)
            print('Beta shape:', self._beta.shape)

    def fit(self, x, y, display_time: bool = False):
        H = self._activation(x.dot(self._w) + self._bias)

        # Moore–Penrose pseudo inverse
        if display_time:
            start = time.time()

        H_pinv = jnp.linalg.pinv(H)

        if display_time:
            stop = time.time()

            print(f'Train time: {stop - start}')

        self._beta = H_pinv.dot(y)

    def __call__(self, x):
        H = self._activation(x.dot(self._w) + self._bias)
        return H.dot(self._beta)

    def evaluate(self, x, y):
        pred = self(x)

        # Loss (base on model setting)
        loss = self._loss(y, pred)

        # Accuracy
        acc = np.sum(np.argmax(pred, axis=-1) == np.argmax(y, axis=-1)) / len(y)

        return loss, acc, pred

    @property
    def beta(self):
        return self._beta

    @property
    def w(self):
        return self._w
