import numpy as np
from utils import sigmoid, softmax
import os

class DecoderLSTM:
    def __init__(self, embedding, hidden_size, vocab_size, learning_rate=1e-2):
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        embedding_dim = embedding.shape[1]

        if os.path.exists("data/model.npz"):
            params = np.load("data/model.npz", allow_pickle=True)
            self.Wx = params["decoder_Wx"]
            self.Wh = params["decoder_Wh"]
            self.b  = params["decoder_b"]
            self.Why  = params["decoder_Why"]
            self.by = params["decoder_by"]
        else:
            # Combine all 4 gates into one big matrix
            self.Wx = np.random.randn(4*hidden_size, embedding_dim)*0.01
            self.Wh = np.random.randn(4*hidden_size, hidden_size)*0.01
            self.b = np.zeros((4*hidden_size, 1))

            self.Why = np.random.randn(vocab_size, hidden_size)*0.01
            self.by = np.zeros((vocab_size, 1))

    def forward(self, inputs, hprev, cprev):
        xs, hs, cs, os, ys, ps = {}, {}, {}, {}, {}, {}
        zs = {}  # pre-activations for gates
        hs[-1], cs[-1] = np.copy(hprev), np.copy(cprev)

        for t, idx in enumerate(inputs):
            xs[t] = self.embedding[idx].reshape(-1, 1)  # column vector

            z = np.dot(self.Wx, xs[t]) + np.dot(self.Wh, hs[t-1]) + self.b
            zs[t] = z

            # split into 4 parts (input, forget, output, candidate)
            i = sigmoid(z[0:self.hidden_size])
            f = sigmoid(z[self.hidden_size:2*self.hidden_size])
            o = sigmoid(z[2*self.hidden_size:3*self.hidden_size])
            g = np.tanh(z[3*self.hidden_size:4*self.hidden_size])

            cs[t] = f * cs[t-1] + i * g
            hs[t] = o * np.tanh(cs[t])
            os[t] = o

            y = np.dot(self.Why, hs[t]) + self.by
            ys[t] = y
            ps[t] = softmax(y)

        return xs, hs, cs, os, zs, ys, ps

    def forward_step(self, token_idx, h_prev, c_prev):
        x_t = self.embedding[token_idx].reshape(-1, 1)

        z = np.dot(self.Wx, x_t) + np.dot(self.Wh, h_prev) + self.b

        i = sigmoid(z[0:self.hidden_size])
        f = sigmoid(z[self.hidden_size:2*self.hidden_size])
        o = sigmoid(z[2*self.hidden_size:3*self.hidden_size])
        g = np.tanh(z[3*self.hidden_size:4*self.hidden_size])

        c_t = f * c_prev + i * g
        h_t = o * np.tanh(c_t)

        y_t = np.dot(self.Why, h_t) + self.by
        p_t = softmax(y_t)

        return h_t, c_t, p_t


    def backward(self, xs, hs, cs, os, zs, ys, ps, targets, inputs):
        dWx, dWh, db = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)
        dWhy, dby = np.zeros_like(self.Why), np.zeros_like(self.by)
        dE = np.zeros_like(self.embedding)

        dhnext = np.zeros((self.hidden_size, 1))
        dcnext = np.zeros((self.hidden_size, 1))
        loss = 0.0

        for t in reversed(range(len(xs))):
            dy = np.copy(ps[t])
            target_idx = targets[t]
            loss -= np.log(max(ps[t][target_idx, 0], 1e-12))
            dy[target_idx, 0] -= 1.0
            
            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            dh = np.dot(self.Why.T, dy) + dhnext
            dc = dcnext + dh * os[t] * (1 - np.tanh(cs[t])**2)

            # unpack gates
            z = zs[t]
            i = sigmoid(z[0:self.hidden_size])
            f = sigmoid(z[self.hidden_size:2*self.hidden_size])
            o = sigmoid(z[2*self.hidden_size:3*self.hidden_size])
            g = np.tanh(z[3*self.hidden_size:4*self.hidden_size])

            di = dc * g
            df = dc * cs[t-1]
            do = dh * np.tanh(cs[t])
            dg = dc * i

            di_raw = di * i * (1 - i)
            df_raw = df * f * (1 - f)
            do_raw = do * o * (1 - o)
            dg_raw = dg * (1 - g*g)

            dz = np.vstack((di_raw, df_raw, do_raw, dg_raw))

            dWx += np.dot(dz, xs[t].T)
            dWh += np.dot(dz, hs[t-1].T)
            db += dz

            dhnext = np.dot(self.Wh.T, dz)
            dcnext = f * dc

            # embedding gradient update
            dE[inputs[t]] += np.dot(self.Wx.T, dz).ravel()

        return dWx, dWh, db, dWhy, dby, dE, dhnext, dcnext, loss

    def update_params(self, grads):
        """
        grads: tuple returned by backward
        Returns dE (the gradient matrix) for convenience (so caller can combine with other grads).
        """

        dWx, dWh, db, dWhy, dby, dE = grads

        self.Wx -= self.learning_rate * dWx
        self.Wh -= self.learning_rate * dWh
        self.b  -= self.learning_rate * db
        self.Why -= self.learning_rate * dWhy
        self.by  -= self.learning_rate * dby

        return dE
        
