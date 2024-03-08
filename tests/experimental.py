import neuro
import neuro.nn as nn


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.lin1 = nn.Linear(4, 5, bias=True)
            self.lin2 = nn.Linear(5, 8, bias=True)

        def forward(self, x):
            return self.lin2(self.lin1(x)).sum()

    model = Model()
    x = neuro.rand((1, 4))
    y = model(x)
    print(y)
    y.backward()
    for n, p in model.namedparams():
        print(n)
        assert p.grad is not None



if __name__ == "__main__":
    main()
