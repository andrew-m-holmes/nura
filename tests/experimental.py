import neuro
import neuro.nn as nn
import numpy as np


def main():

    # class Model(nn.Module):
    #
    #     def __init__(self) -> None:
    #         super().__init__()
    #         self.lin1 = nn.Linear(10, 10)
    #         self.lin2 = nn.Linear(10, 10)
    #         self.lin3 = nn.Linear(10, 10)
    #         self.lin3.lin = nn.Linear(10, 10, bias=False)
    #
    # model = Model()

    param = nn.parameter(neuro.empty(0), neuro.float)
    print(param.to(neuro.double))
    param.mutate(usegrad=False)
    print(type(param.mutated(usegrad=True)))


if __name__ == "__main__":
    main()
