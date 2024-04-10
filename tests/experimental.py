import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    x = nura.rand(3, 4, usegrad=True)
    g = nura.rand(4, usegrad=True)
    b = nura.rand(4, usegrad=True)
    y = f.layernorm(x, g, b, dim=-1)
    loss = y.sum()
    loss.backward()
    print(x.grad)
    print(g.grad)
    print(b.grad)


if __name__ == "__main__":
    main()
