import nura
import nura.nn as nn
import numpy as np


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 8)

        def forward(self, x):
            return self.linear(x)

    model = Model()
    modelhf = model.half()
    modelfl = model.float()
    modeldf = model.double()
    print(model, modelhf, modelfl, modeldf, sep="\n\n")
    print(modelhf is modelfl)
    print(modelhf is modeldf)
    print(modelhf is model)


if __name__ == "__main__":
    main()
