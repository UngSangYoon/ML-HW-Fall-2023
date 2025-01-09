import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt


#  nn.Module을 상속하는 클래스 생성
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        # affine layer 생성
        l1 = nn.Linear(input_size, hidden_size, bias=True)
        l2 = nn.Linear(hidden_size, hidden_size // 2, bias=True)
        l3 = nn.Linear(hidden_size // 2, hidden_size // 4, bias=True)
        l4 = nn.Linear(hidden_size // 4, hidden_size // 8, bias=True)
        l5 = nn.Linear(hidden_size // 8, output_size, bias=True)
        # BatchNormalization layer 생성
        bn1 = torch.nn.BatchNorm1d(hidden_size)
        bn2 = torch.nn.BatchNorm1d(hidden_size // 2)
        bn3 = torch.nn.BatchNorm1d(hidden_size // 4)
        bn4 = torch.nn.BatchNorm1d(hidden_size // 8)
        # He initialization으로 가중치 초기화
        nn.init.kaiming_uniform_(l1.weight)
        nn.init.kaiming_uniform_(l2.weight)
        nn.init.kaiming_uniform_(l3.weight)
        nn.init.kaiming_uniform_(l4.weight)
        nn.init.kaiming_uniform_(l5.weight)
        # 활성화 함수로 LeakyReLU 사용
        activation = torch.nn.LeakyReLU()

        # layer 합치기
        self.model = nn.Sequential(
            l1,
            bn1,
            activation,
            l2,
            bn2,
            activation,
            l3,
            bn3,
            activation,
            l4,
            bn4,
            activation,
            l5,
        )


# 학습 코드
def train(loss_list, loss_interval, model, train_loader, optimizer, epoch):
    model.train()
    for i, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(input)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if i % loss_interval == 0:
            loss_list.append(loss.item())
            print(f"epoch: {epoch}  iter: {i}  loss: {loss.item()}")


# 테스트 코드
def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output = model(input)
            predict = output.argmax(dim=1, keepdim=True)
            real = target.argmax(dim=1, keepdim=True)  # one-hot vector이기 때문에 고쳐줌
            correct += predict.eq(real).sum().item()
    print(f"accuracy: {correct/len(test_loader.dataset)}")
    return correct / len(test_loader.dataset)


# 실행 코드
def nn_pytorch_run(
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
    # set hyperparameters
    model = NeuralNet(input_size, hidden_size, output_size).model
    batch_size = batch_size
    learning_rate = learning_rate
    # dataset을 PyTorch tensor로 바꿈
    x_train, t_train, x_test, t_test = map(
        torch.tensor, (x_train, t_train, x_test, t_test)
    )
    train_dataset = TensorDataset(x_train, t_train)
    test_dataset = TensorDataset(x_test, t_test)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # set optimizer(Adam or SGD)
    if Use_Adam:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    loss_list = []

    # training
    for i in range(epoch):
        train(loss_list, loss_interval, model, train_loader, optimizer, epoch=i + 1)
        acc = test(model, test_loader)

    # show loss
    if Use_Adam == True:
        plt.plot(loss_list)
        plt.title(f"Neural Net Using Pytorch Loss(Adam)  accuracy: {acc}")
        plt.show
        plt.savefig("Neural Net Using Pytorch Loss(Adam).png")
    else:
        plt.plot(loss_list)
        plt.title(f"Neural Net Using Pytorch Loss(SGD) accuracy: {acc}")
        plt.show
        plt.savefig("Neural Net Using Pytorch Loss(SGD).png")
