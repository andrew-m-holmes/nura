import nura
import nura.nn as nn
import torch
import torch.nn.functional as f
import numpy as np


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 8)

        def forward(self, x):
            return self.linear(x)

    a = nura.tensor([-2.0, 3.0, 0.0]).usedgrad()
    b = nura.tensor(2.0).usedgrad()
    c = a**b
    print(c)
    c.backward(nura.oneslike(c))
    print(a.grad)
    print(b.grad)


if __name__ == "__main__":
    main()
