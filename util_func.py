import numpy as np


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# softmax function
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x


# batched mean squared error
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2) / y.shape[0]


# batched cross entropy error
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # target : one-hot vector
    return -np.sum(t * np.log(y + delta)) / batch_size


# random initialization
def random_init(n_in, n_out):
    return np.random.randn(n_in, n_out)


# Xavier initialization
def xavier_init(n_in, n_out):
    return np.random.randn(n_in, n_out) / np.sqrt(n_in)


# He initialization
def he_init(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)


# adam optimizer
def adam(grad, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad  # first moment
    v = beta2 * v + (1 - beta2) * (grad**2)  # second moment
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    grad_update = m_hat / (np.sqrt(v_hat) + eps)
    return m, v, grad_update  # return updated m, v, and grad
