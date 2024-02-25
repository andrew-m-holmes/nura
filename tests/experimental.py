import numpy as np
import deepnet
import jax
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd, getperts
import torch


def main():

    a = deepnet.rand((5, 4), usegrad=True).float()
    b = a.mutated(grad=deepnet.oneslike(a))
    with deepnet.autograd(reverse=False, forward=True):
        c = deepnet.min(b, dim=(0, 1))
        print(c.grad)


if __name__ == "__main__":
    main()
