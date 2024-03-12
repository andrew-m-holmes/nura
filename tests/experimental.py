import nura
import nura.nn as nn
import torch
import torch.nn as tnn


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.lin1 = nn.Linear(4, 5, bias=True)
            self.relu = nn.ReLU()
            self.lin2 = nn.Linear(5, 8, bias=True)
            self.softmax = nn.Softmax()

        def forward(self, x):
            x = self.relu(self.lin1(x))
            out = self.softmax(self.lin2(x))
            return out

    model = Model()
    x = nura.rand((1, 4))
    out = model(x)
    loss = out.sum() / nura.rand()
    loss.backward()
    print(loss)


if __name__ == "__main__":
    main()
