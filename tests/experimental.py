import nura
import nura.nn as nn
import torch
import torch.nn as tnn
import numpy as np


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = self.linear(3, 5, bias=True)
            self.fc2 = self.linear(5, 3, bias=True)
            self.fc3 = self.linear(3, 1, bias=False)
            self.relu = nn.ReLU()
            self.sig = nn.Sigmoid()

        def forward(self, x: nura.Tensor):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            out = self.sig(self.fc3(x))
            return out

    model = Model()
    x = nura.rand((1, 3))
    out = model(x)
    out.backward()
    print(model)


if __name__ == "__main__":
    main()
