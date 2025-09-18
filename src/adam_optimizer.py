import numpy as np

class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}

    def _init_state(self, name, param):
        if name not in self.m:
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    def update(self, name, param, grad, t):
        self._init_state(name, param)

        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad * grad)

        m_hat = self.m[name] / (1 - self.beta1 ** t)
        v_hat = self.v[name] / (1 - self.beta2 ** t)

        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param

    def update_embedding(self, name, embedding, dE, used_idxs, t):
        self._init_state(name, embedding)
        
        for idx in used_idxs:
            grad = dE[idx]

            self.m[name][idx] = self.beta1 * self.m[name][idx] + (1 - self.beta1) * grad
            self.v[name][idx] = self.beta2 * self.v[name][idx] + (1 - self.beta2) * (grad * grad)

            m_hat = self.m[name][idx] / (1 - self.beta1 ** t)
            v_hat = self.v[name][idx] / (1 - self.beta2 ** t)

            embedding[idx] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return embedding