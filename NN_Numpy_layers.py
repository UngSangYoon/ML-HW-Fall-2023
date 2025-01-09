from util_func import *


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] *= self.alpha
        return out

    def backward(self, dout):
        dout[self.mask] *= self.alpha
        return dout


# 'SiLU' is also known as 'swish'
class SiLU:
    def __init__(self):
        self.sigmoid = None
        self.out = None

    def forward(self, x):
        self.sigmoid = sigmoid(x)
        self.out = x * self.sigmoid
        return self.out

    def backward(self, dout):
        return dout * (self.out + self.sigmoid * (1 - self.out))


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.dW = None
        self.db = None

        self.dW_m = np.zeros_like(W)
        self.db_m = np.zeros_like(b)
        self.dW_v = np.zeros_like(W)
        self.db_v = np.zeros_like(b)

        self.t = 0

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


# Mean Squared Error Loss for batched input
class MSELoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = x
        self.loss = mean_squared_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


# batch normalization
class BatchNorm:
    def __init__(self, dims):
        self.gamma = np.ones((1, dims), dtype=np.float32)
        self.beta = np.zeros((1, dims), dtype=np.float32)
        self.x = None
        self.dgamma = None
        self.dbeta = None
        self.dx = None
        self.mu = None
        self.var = None
        self.x_hat = None
        self.batch_size = None
        self.epsilon = 1e-7

        self.dgamma_m = np.zeros_like(self.gamma)
        self.dbeta_m = np.zeros_like(self.beta)
        self.dgamma_v = np.zeros_like(self.gamma)
        self.dbeta_v = np.zeros_like(self.beta)

        self.t = 0

    def forward(self, x):
        self.x = x
        self.batch_size = self.x.shape[0]
        self.mu = np.mean(self.x, axis=0, keepdims=True)
        self.var = np.var(self.x, axis=0, keepdims=True)
        self.x_hat = (self.x - self.mu) / np.sqrt(self.var + self.epsilon)
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, dout):
        self.dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        dx_hat = dout * self.gamma
        x_mu = self.x - self.mu
        var_eps = self.var + self.epsilon
        dvar = np.sum(
            dx_hat * x_mu * (-0.5) * (var_eps ** (-1.5)), axis=0, keepdims=True
        )
        dmu = np.sum(
            dx_hat * (-1) / np.sqrt(var_eps), axis=0, keepdims=True
        ) - dvar * np.mean(2 * x_mu, axis=0, keepdims=True)
        self.dx = (
            (dx_hat / np.sqrt(var_eps))
            + (dvar * 2 * x_mu / self.batch_size)
            + (dmu / self.batch_size)
        )
        return self.dx
