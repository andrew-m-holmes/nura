import nura
import nura.nn as nn
import nura.nn.utils as u
import nura.nn.functional as f
import numpy as np


def main():

    x = nura.randn(2, 3, 2, 2, 4).attach()
    m = nn.BatchNorm(4)
    m.train()
    print(m)
    y = m(x)
    print(y)
    z = y.sum()
    z.backward()
    print(m.gamma.grad)
    print(m.beta.grad)
    print(m._mean)
    print(m._var)


if __name__ == "__main__":
    main()
