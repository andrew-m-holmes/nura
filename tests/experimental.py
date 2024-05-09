import numpy as np
import nura
import nura.nn as nn


class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.softmax(self.linear3(x))
        return x


def main():

    model = Model()
    x = nura.randn(64, 784)
    y = nura.randint(0, 10, (64,))
    output = model(x)
    objective = nn.CrossEntropy(ignoreid=5)
    loss = objective(output, y)
    print(loss)
    loss.backward()


if __name__ == "__main__":
    main()
