import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    w = nn.parameter(nura.randn(4, 5))
    b = nn.parameter(nura.randn(4))
    x = nura.randn(2, 5)
    y = nura.randint(0, 4, 2)
    rmsprop = nn.RMSProp(iter([w, b]), learnrate=0.1, alpha=0.9, decay=1)
    o = f.linear(x, w, b)
    loss = f.crossentropy(o, y)
    print(loss)
    loss.backward()
    rmsprop.step()
    rmsprop.zerograd()
    for m in rmsprop.moments():
        print(m)
    print(rmsprop)


if __name__ == "__main__":
    main()
