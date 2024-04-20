import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    w = nn.parameter(nura.randn(4, 5))
    b = nn.parameter(nura.randn(4))
    x = nura.randn(2, 5)
    y = nura.randint(0, 4, 2)
    sgd = nn.SGD(iter([w, b]), 1e-2, momentum=0.9, nesterov=True, decay=0)
    o = f.linear(x, w, b)
    loss = f.crossentropy(o, y)
    print(loss)
    loss.backward()
    sgd.step()
    sgd.zerograd()


if __name__ == "__main__":
    main()
