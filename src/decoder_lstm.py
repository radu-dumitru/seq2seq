import numpy as np
from utils import sigmoid, softmax

class DecoderLSTM:
    def __init__(self, embedding, hidden_size, vocab_size, learning_rate=1e-2):
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        embedding_dim = embedding.shape[1]

        # Combine all 4 gates into one big matrix
        self.Wx = np.random.randn(4*hidden_size, embedding_dim)*0.01
        self.Wh = np.random.randn(4*hidden_size, hidden_size)*0.01
        self.b = np.zeros((4*hidden_size, 1))

        # encourage remembering at the start
        self.b[self.hidden_size:2*self.hidden_size, :] = 1.0

        self.Why = np.random.randn(vocab_size, hidden_size)*0.01
        self.by = np.zeros((vocab_size, 1))
        
        # internal cache for backward
        self._cache = None
        
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)
        self.dE = np.zeros_like(self.embedding)

    def forward(self, inputs, state):
        hprev, cprev = state
        xs, hs, cs, os, ys = {}, {}, {}, {}, {}
        ps = {}
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

        self._cache = {
            "xs": xs, "hs": hs, "cs": cs, "os": os, "zs": zs,
            "ys": ys, "ps": ps, "inputs": inputs
        }

        return (hs[len(xs) - 1], cs[len(xs) - 1])

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

    def backward(self, targets):
        xs = self._cache["xs"]
        hs = self._cache["hs"]
        cs = self._cache["cs"]
        os = self._cache["os"]
        zs = self._cache["zs"]
        ys = self._cache["ys"]
        ps = self._cache["ps"]
        inputs = self._cache["inputs"]

        self.dWx.fill(0.0)
        self.dWh.fill(0.0)
        self.db.fill(0.0)
        self.dWhy.fill(0.0)
        self.dby.fill(0.0)
        self.dE.fill(0.0)

        dhnext = np.zeros((self.hidden_size, 1))
        dcnext = np.zeros((self.hidden_size, 1))
        loss = 0.0

        for t in reversed(range(len(xs))):
            dy = np.copy(ps[t])
            target_idx = targets[t]
            loss -= np.log(max(ps[t][target_idx, 0], 1e-12))
            dy[target_idx, 0] -= 1.0
            
            self.dWhy += np.dot(dy, hs[t].T)
            self.dby += dy

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

            self.dWx += np.dot(dz, xs[t].T)
            self.dWh += np.dot(dz, hs[t-1].T)
            self.db += dz

            dhnext = np.dot(self.Wh.T, dz)
            dcnext = f * dc

            # embedding gradient update
            self.dE[inputs[t]] += np.dot(self.Wx.T, dz).ravel()

        return dhnext, dcnext, loss

    def parameters(self):
        return [
            ("decoder_Wx", self.Wx, self.dWx),
            ("decoder_Wh", self.Wh, self.dWh),
            ("decoder_b",  self.b,  self.db),
            ("decoder_Why", self.Why, self.dWhy),
            ("decoder_by",  self.by,  self.dby),
        ]

    def state_dict(self):
        return {
            "Wx": self.Wx.copy(),
            "Wh": self.Wh.copy(),
            "b": self.b.copy(),
            "Why": self.Why.copy(),
            "by": self.by.copy(),
        }

    def load_state_dict(self, state):
        self.Wx[...] = state["Wx"]
        self.Wh[...] = state["Wh"]
        self.b[...] = state["b"]
        self.Why[...] = state["Why"]
        self.by[...] = state["by"]
        
