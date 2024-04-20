import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    x = nura.randn(3, 7)
    x.usesgrad()
    y = nura.randint(0, 7, 3)
    ignoreid = int(nura.randint(0, 7).item())
    loss = f.crossentropy(x, y, ignoreid)
    loss.backward()
    print(x.grad)


if __name__ == "__main__":
    main()
