import neuro
import neuro.nn as nn


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.lin1 = nn.Linear(4, 5, bias=True)
            self.lin2 = nn.Linear(5, 8, bias=True)
            self.lin2.lin = nn.Linear(4, 5)
            self.lst = []

        def forward(self, x):
            return self.lin2(self.lin1(x)).sum()

    model = Model()
    hfmodel = model.half()
    fpmodel = model.float()
    dbmodel = model.double()
    print(model)
    print(hfmodel)
    print(fpmodel)
    print(dbmodel)
    model.lst.append(10)
    print(f"{hfmodel.lst is dbmodel.lst = }")
    print(f"{fpmodel.lin1 is model.lin1 = }")
    print(f"{fpmodel.lin2.lin is dbmodel.lin2.lin = }")


if __name__ == "__main__":
    main()
