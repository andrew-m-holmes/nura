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
            self.bro = nn.Buffer()

    model = Model()
    print(model)
    model.float()
    for _, p in model.iterparams():
        print(p.dtype)
    model.double()
    for _, p in model.iterparams():
        print(p.dtype)
    model.half()
    for _, p in model.iterparams():
        print(p.dtype)

if __name__ == "__main__":
    main()
