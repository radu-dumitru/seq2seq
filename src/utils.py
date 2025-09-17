import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)