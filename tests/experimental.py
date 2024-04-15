import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    p = nn.parameter(nura.rand(4))
    parameters = iter([p])
    sgd = nn.SGD(parameters, 1e-2)
    y = p / 3.0
    loss = y.sum()
    loss.backward()
    print(p)
    sgd.step()
    print(p)


if __name__ == "__main__":
    main()
