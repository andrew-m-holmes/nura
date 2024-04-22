import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    w = nn.parameter(nura.randn(4, 5))
    b = nn.parameter(nura.randn(4))
    x = nura.randn(2, 5)
    y = nura.randint(0, 4, 2)
    adam = nn.Adam(iter([w, b]), learnrate=0.1, betas=(0.9, 0.99), decay=1)
    o = f.linear(x, w, b)
    loss = f.crossentropy(o, y)
    print(w)
    loss.backward()
    adam.step()
    adam.zerograd()
    print(w)


if __name__ == "__main__":
    main()
