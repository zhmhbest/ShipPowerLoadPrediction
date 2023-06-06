import numpy as np


def rmse_nd(yp: np.ndarray, yt: np.ndarray):
    return np.sqrt(np.mean(np.power(yt - yp, 2.0))) / np.mean(yt) * 100


def mape_nd(yp: np.ndarray, yt: np.ndarray):
    yp += 1
    yt += 1
    return np.mean(np.abs((yt - yp) / yt)) * 100
