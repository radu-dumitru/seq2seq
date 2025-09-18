import numpy as np
from utils import sigmoid, softmax

class EncoderLSTM:
    def __init__(self, embedding, hidden_size, learning_rate=1e-2):
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        embedding_dim = embedding.shape[1]
        
        # Combine all 4 gates into one big matrix
        self.Wx = np.random.randn(4*hidden_size, embedding_dim)*0.01
        self.Wh = np.random.randn(4*hidden_size, hidden_size)*0.01
        self.b = np.zeros((4*hidden_size, 1))
            
    def forward(self, inputs, hprev, cprev):
        xs, hs, cs, os = {}, {}, {}, {}
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

        return xs, hs, cs, os, zs

    def backward(self, xs, hs, cs, os, zs, dhnext, dcnext, inputs):
        dWx, dWh, db = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)
        dE = np.zeros_like(self.embedding)

        for t in reversed(range(len(xs))):
            
            dh = dhnext
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

        return dWx, dWh, db, dE, dhnext, dcnext

