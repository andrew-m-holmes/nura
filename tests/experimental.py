import numpy as np
import deepnet
import jax
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd, getperts
import torch


def main():
    
    a = np.random.rand(3, 3, 2)
    b = np.max(a, axis=2)
    print(a == b)

    
    a = deepnet.rand(2)
    b = deepnet.rand(2)
    print(a > b, a == b, a >= b, a < b, a <= b, a != b, not a, a and b, a or b, sep="\n")


if __name__ == "__main__":
    main()
