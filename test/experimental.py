import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(768, 512)
        self.norm = nn.LayerNorm(512)
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        x = f.relu(self.norm(self.linear1(x)))
        return self.linear2(x)


def main():

    model = Model()
    lossfn = nn.CrossEntropy()
    optimizer = nn.AdaGrad(model.parameters(), learnrate=1.0)

    inputs = nura.randn(2, 768)
    labels = nura.randint(0, 10, 2)

    outputs = model(inputs)
    loss = lossfn(outputs, labels)
    loss.backward()

    optimizer.step()
    print(optimizer)


if __name__ == "__main__":
    main()
