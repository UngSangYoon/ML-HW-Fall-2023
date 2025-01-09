from util_func import *
from NN_Numpy_layers import *
from matplotlib import pyplot as plt


class NeuralNet:
    def __init__(
        self,
        x_train,
        t_train,
        x_test,
        t_test,
        input_size,
        hidden_size,
        output_size,
        learning_rate,
        Use_Adam,
    ):
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.Use_Adam = Use_Adam
        # 가중치 초기화
        self.learning_rate = learning_rate
        params = {}
        params["W1"] = he_init(input_size, 256)
        params["b1"] = np.zeros(hidden_size)
        params["W2"] = he_init(hidden_size, hidden_size // 2)
        params["b2"] = np.zeros(hidden_size // 2)
        params["W3"] = he_init(hidden_size // 2, hidden_size // 4)
        params["b3"] = np.zeros(hidden_size // 4)
        params["W4"] = he_init(hidden_size // 4, hidden_size // 8)
        params["b4"] = np.zeros(hidden_size // 8)
        params["W5"] = he_init(hidden_size // 8, output_size)
        params["b5"] = np.zeros(output_size)

        # Build layers
        self.layers = [
            Affine(params["W1"], params["b1"]),
            BatchNorm(hidden_size),
            LeakyReLU(),
            Affine(params["W2"], params["b2"]),
            BatchNorm(hidden_size // 2),
            LeakyReLU(),
            Affine(params["W3"], params["b3"]),
            BatchNorm(hidden_size // 4),
            LeakyReLU(),
            Affine(params["W4"], params["b4"]),
            BatchNorm(hidden_size // 8),
            LeakyReLU(),
            Affine(params["W5"], params["b5"]),
        ]

        self.last_layer = SoftmaxWithLoss()

        self.loss_list = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, input, target):
        output = self.forward(input)
        loss = self.last_layer.forward(output, target)
        return loss

    def backward(self, input, target):
        self.loss(input, target)
        dout = 1
        dout = self.last_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return

    def update_params(self):
        for layer in self.layers:
            if isinstance(layer, Affine):
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db
            elif isinstance(layer, BatchNorm):
                layer.gamma -= self.learning_rate * layer.dgamma
                layer.beta -= self.learning_rate * layer.dbeta

    def update_params_adam(self):
        for layer in self.layers:
            if isinstance(layer, Affine):
                layer.t += 1
                layer.dW_m, layer.dW_v, layer.dW = adam(
                    layer.dW, layer.dW_m, layer.dW_v, layer.t
                )
                layer.db_m, layer.db_v, layer.db = adam(
                    layer.db, layer.db_m, layer.db_v, layer.t
                )
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db
            elif isinstance(layer, BatchNorm):
                layer.t += 1
                layer.dgamma_m, layer.dgamma_v, layer.dgamma = adam(
                    layer.dgamma, layer.dgamma_m, layer.dgamma_v, layer.t
                )
                layer.dbeta_m, layer.dbeta_v, layer.dbeta = adam(
                    layer.dbeta, layer.dbeta_m, layer.dbeta_v, layer.t
                )
                layer.gamma -= self.learning_rate * layer.dgamma
                layer.beta -= self.learning_rate * layer.dbeta

    def train(
        self,
        epoch,
        batch_size,
        loss_interval,
    ):
        input = self.x_train
        target = self.t_train
        data_size = input.shape[0]
        epoch_iter = len(input) // batch_size
        total_iter = epoch * epoch_iter
        for i in range(epoch):
            if i != 0:
                self.test(self.x_test, self.t_test)
            for j in range(i * epoch_iter, i * epoch_iter + epoch_iter):
                batch_mask = np.random.choice(data_size, batch_size)
                input_batch = input[batch_mask]
                target_batch = target[batch_mask]
                self.backward(input_batch, target_batch)
                if self.Use_Adam:
                    self.update_params_adam()
                else:
                    self.update_params()
                if j % loss_interval == 0:
                    loss = self.loss(input_batch, target_batch)
                    self.loss_list.append(loss)
                    print(f"epoch: {i +1}  iter: {j % epoch_iter}  loss: {loss}")

    def predict(self, input):
        output = self.forward(input)
        output = np.argmax(output, axis=1)
        return output

    def test(self, input, target):
        correct = 0
        predict_num = self.predict(input)
        real_num = np.argmax(target, axis=1)
        for i in range(len(real_num)):
            if predict_num[i] == real_num[i]:
                correct += 1
        acc = correct / len(real_num)
        print(f"accuracy: {acc}")
        return acc


def nn_numpy_run(
    x_train,
    t_train,
    x_test,
    t_test,
    input_size,
    hidden_size,
    output_size,
    learning_rate,
    epoch,
    batch_size,
    loss_interval,
    Use_Adam,
):
    nn = NeuralNet(
        x_train,
        t_train,
        x_test,
        t_test,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate,
        Use_Adam=Use_Adam,
    )
    nn.train(
        epoch,
        batch_size,
        loss_interval,
    )
    acc = nn.test(x_test, t_test)

    # show loss
    if Use_Adam:
        plt.clf
        plt.plot(nn.loss_list)
        plt.title(f"Neural Net Using Numpy Loss(Adam)  accuracy: {acc}")
        plt.show()
        plt.savefig("Neural Net Using Numpy Loss(Adam).png")
    else:
        plt.clf
        plt.plot(nn.loss_list)
        plt.title(f"Neural Net Using Numpy Loss(SGD)   accuracy: {acc}")
        plt.show()
        plt.savefig("Neural Net Using Numpy Loss(SGD).png")
