import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    z = (nura.randn(4, 1) - 0.5).usedgrad()
    a = f.sigmoid(z)
    y = nura.tensor([1, 0, 1, 0]).float()
    lossfn = nn.BinaryCrossEntropy()
    loss = lossfn(a, y)
    loss.backward()
    print(lossfn)
    print(z.grad)


if __name__ == "__main__":
    main()
