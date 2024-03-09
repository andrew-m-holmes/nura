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

    x = neuro.randint(-5, 5, (1, 3), dtype=neuro.float).mutated(usegrad=True)
    z = nn.relu(x)
    print(z)
    w = nn.sigmoid(x)
    print(w)
    y = nn.tanh(x)
    print(y)
    a = nn.softmax(x)
    print(a)
    z.backward(neuro.oneslike(z))
    print(x.grad)

if __name__ == "__main__":
    main()
