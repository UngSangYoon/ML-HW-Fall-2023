from NN_Numpy import *
from NN_Pytorch import *
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(
    flatten=True, normalize=True, one_hot_label=True
)

nn_numpy_run(
    x_train,
    t_train,
    x_test,
    t_test,
    input_size=784,
    hidden_size=256,
    output_size=10,
    learning_rate=0.001,
    epoch=10,
    batch_size=100,
    loss_interval=100,
    Use_Adam=False,
)
"""
nn_numpy_run(
    x_train,
    t_train,
    x_test,
    t_test,
    input_size=784,
    hidden_size=256,
    output_size=10,
    learning_rate=0.0001,
    epoch=10,
    batch_size=100,
    loss_interval=100,
    Use_Adam=True,
)

nn_pytorch_run(
    x_train,
    t_train,
    x_test,
    t_test,
    input_size=784,
    hidden_size=256,
    output_size=10,
    learning_rate=0.001,
    epoch=10,
    batch_size=100,
    loss_interval=100,
    Use_Adam=False,
)

nn_pytorch_run(
    x_train,
    t_train,
    x_test,
    t_test,
    input_size=784,
    hidden_size=256,
    output_size=10,
    learning_rate=0.0001,
    epoch=10,
    batch_size=100,
    loss_interval=100,
    Use_Adam=True,
)
"""
