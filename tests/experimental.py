import nura
import nura.nn as nn
import torch


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 8)

        def forward(self, x):
            return self.linear(x)

    model = Model()
    x = nura.randn().usesgrad()
    a = nn.relu(x).backward()
    b = nn.relu6(x).backward()
    c = nn.leakyrelu(x).backward()
    d = nn.gelu(x).backward()


if __name__ == "__main__":
    main()
