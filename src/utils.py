import numpy as np
from constants import UNK_TOKEN

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    # x can be (vocab, 1) or (vocab,) — make it 2D and operate on axis=0 safely
    x = np.asarray(x)
    # keep dims so broadcasting is safe for column vectors
    x = x - np.max(x, axis=0, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=0, keepdims=True)


def tokenize(text):
    return text.split()

def words_to_indices(words, word2idx):
    return [word2idx.get(w, word2idx[UNK_TOKEN]) for w in words]

def clip_grad_norm_(params, max_norm, eps=1e-12):
    total = 0.0
    for _, _, g in params:
        if g is None:
            continue
        total += float(np.sum(g * g))
    norm = np.sqrt(total)
    if norm > max_norm and norm > 0:
        scale = max_norm / (norm + eps)
        for _, _, g in params:
            if g is not None:
                g *= scale
    return norm

def save_model_params(embedding, encoder, decoder, optimizer, step, path="data/model.npz"):
    np.savez(
        path,
        embedding=embedding,
        encoder_state=encoder.state_dict(),
        decoder_state=decoder.state_dict(),
        optimizer_state=optimizer.state_dict(),
        step=step
    )

def load_model_params(encoder, decoder, optimizer, path="data/model.npz"):
    import os
    if os.path.exists(path):
        params = np.load(path, allow_pickle=True)
        embedding = params["embedding"]
        encoder.load_state_dict(params["encoder_state"].item())
        decoder.load_state_dict(params["decoder_state"].item())
        optimizer.load_state_dict(params["optimizer_state"].item())
        step = int(params["step"])
        return embedding, step
    else:
        return None, 0