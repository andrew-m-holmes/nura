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

    a = neuro.rand((3, 4), usegrad=True).float()
    y = nn.sigmoid(a)
    print(y)


if __name__ == "__main__":
    main()
