import numpy as np
import torch


def concordance_correlation_coefficient(y_true, y_pred, eps=1e-8):

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    vx = y_true - mean_true
    vy = y_pred - mean_pred
    cor = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = sd_true**2 + sd_pred**2 + (mean_true - mean_pred) ** 2

    return numerator / denominator

