import nura
import nura.nn as nn
from nura.autograd.functional import jacrev, jacfwd
import numpy as np


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(3, 5, bias=True)
            self.fc2 = nn.Linear(5, 3, bias=True)
            self.fc3 = nn.Linear(3, 1, bias=False)
            self.relu = nn.ReLU()
            self.sig = nn.Sigmoid()

        def forward(self, x: nura.Tensor):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            out = self.sig(self.fc3(x))
            return out


if __name__ == "__main__":
    main()
