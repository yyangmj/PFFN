import numpy as np
from torch import nn
import torch

def score(y_predict, y_real):
    a1 = 13
    a2 = 10
    y_real = y_real.numpy()
    error = (y_predict - y_real)*0.01
    #print("error array:", error)
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)