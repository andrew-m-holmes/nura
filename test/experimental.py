import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        return self.out(self.relu(self.linear(x)))


def main():

    model = Model()
    lossfn = nn.CrossEntropy()
    optim = nn.AdaDelta(model.parameters())
    print(optim)

    inputs = nura.randn(3, 768)
    labels = nura.randint(0, 10, 3)
    outputs = model(inputs)
    loss = lossfn(outputs, labels)
    loss.backward()
    optim.step()


if __name__ == "__main__":
    main()
