import nura
import nura.nn as nn
import nura.nn.utils as u
import nura.nn.functional as f
import numpy as np


def main():

    a = nura.randn(32, 10, usegrad=True)
    gamma = nura.ones(10)
    beta = nura.zeros(10)
    out = f.batchnorm1d(a, gamma, beta)
    print(out)

    b = nura.randn(32, 7, 64, usegrad=True)
    gamma = nura.ones(64)
    beta = nura.zeros(64)
    out = f.batchnorm1d(b, gamma, beta)
    print(out)


if __name__ == "__main__":
    main()
