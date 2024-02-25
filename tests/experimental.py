import numpy as np
import deepnet
import jax
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd, getperts
import torch


def main():

    a = deepnet.rand((64, 3, 5, 4), usegrad=True).float()
    b = deepnet.rand((4, 3), usegrad=True).float()
    c = deepnet.matmul(a, b)
    c.backward(deepnet.oneslike(c))

    a = np.random.rand(4, 3)
    b = np.random.rand(3)
    c = np.dot(a, b)
    g = np.ones_like(c)
    agrad = np.dot(np.expand_dims(g, -1), np.expand_dims(b, 0))
    print(agrad.shape)

    a = np.random.rand(3, 1)
    b = a.T
    c = np.dot(a, b)
    print(c.shape)

if __name__ == "__main__":
    main()
