import numpy as np


def log_softmax_batch(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return np.log(x / np.sum(x, axis=-1, keepdims=True))
