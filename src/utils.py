import numpy as np
from constants import UNK_TOKEN

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def tokenize(text):
    return text.split()

def words_to_indices(words, word2idx):
    return [word2idx.get(w, word2idx[UNK_TOKEN]) for w in words]