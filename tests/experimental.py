import numpy as np
import deepnet
import jax
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd, getperts
import torch


def main():

    a = deepnet.rand((5, 4, 3), usegrad=True).float()
    b = a.max()
    b.backward()
    print(b)
    c = a.min()
    print(c)
    c.backward()
    print(a.grad)

if __name__ == "__main__":
    main()
