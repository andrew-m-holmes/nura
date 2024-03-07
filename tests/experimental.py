import neuro
import neuro.nn as nn
import numpy as np


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.lin1 = nn.Linear(10, 10)
            self.lin2 = nn.Linear(10, 10)
            self.lin3 = nn.Linear(10, 10)
            self.lin3.lin = nn.Linear(10, 10, bias=False)

    model = Model()
    for n, m in model.modules():
        print(n, m.name())
    for n, p in model.parameters():
        print(n, p)
    model.double()

    for n, p in model.parameters():
        print(n, p.dtype)

if __name__ == "__main__":
    main()
