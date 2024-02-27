import numpy as np
import deepnet
import jax
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd, getperts
import torch
import deepnet.nn as nn


def main():

    mod = nn.Module()
    tensor = deepnet.tensor(5.0, usegrad=False)
    param = nn.Parameter(tensor)

    class Linear(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.weight = param

    linear = Linear()
    print(linear.params())


if __name__ == "__main__":
    main()
