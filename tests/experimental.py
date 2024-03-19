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

    tensor = nura.rand(1, 2, 3).usedgrad()
    print(tensor)


if __name__ == "__main__":
    main()
