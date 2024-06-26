import nura
import nura.nn as nn
import nura.nn.utils as u
import nura.nn.functional as f
import numpy as np


def main():

    a = nura.randn(3, 3, 4, usegrad=True)
    b = nura.randn(3, 3, 1)
    c = nura.concat(a, b, dim=-1)
    print(c.dim)
    c.backward(nura.oneslike(c))
    print(a.grad)


if __name__ == "__main__":
    main()
