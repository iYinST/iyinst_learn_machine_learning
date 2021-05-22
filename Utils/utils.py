import numpy as np
from sklearn.metrics import mean_squared_error
import  time

def se_loss(target: np.array, pred = None):
    if pred:
        return mean_squared_error(target,pred)
    else:
        return np.var(target) * len(target)

def gini_loss(y: np.array):
    gini = 0
    for target in np.unique(y):
        gini += np.sum(y == target) ** 2
    return 1 - gini / len(y) ** 2

def time_count(func):
    def wrapTheFunction(*args, **kwargs):
        t = time.time()

        result = func(*args, **kwargs)

        print("[time count][{}] {}s.".format(func.__name__,time.time() - t))
        return result
    return wrapTheFunction

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


