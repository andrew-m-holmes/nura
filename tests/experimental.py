import nura
import nura.nn as nn


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.lin1 = nn.Linear(4, 5, bias=True)
            self.lin2 = nn.Linear(5, 8, bias=True)

        def forward(self, x):
            return self.lin2(self.lin1(x)).sum()

    model = Model()
    test = model.mutated(training=False)
    print(test.training)


if __name__ == "__main__":
    main()
