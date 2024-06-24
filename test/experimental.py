import nura
import nura.nn as nn
import nura.nn.utils as u
import nura.nn.functional as f
import numpy as np


def main():

    x = nura.randn(2, 3, 10, 5, 4).attach()
    gamma = nura.ones(4).attach()
    beta = nura.zeroslike(gamma).attach()
    y = f.batchnorm(x, gamma, beta)
    y.backward(nura.oneslike(x))
    print(y)
    print(x.grad)
    print(gamma.grad)
    print(beta.grad)


if __name__ == "__main__":
    main()
