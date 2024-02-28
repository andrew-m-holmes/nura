import numpy as np
import deepnet
import jax
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd, getperts
import torch
import deepnet.nn as nn


def main():

    class Linear(nn.Module):
        pass

    class ReLU(nn.Module):
        pass

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.linear = Linear()
            self.relu = ReLU()

    model = Model()
    print(list(model.mods()))
    print(list(model.params()))


if __name__ == "__main__":
    main()
