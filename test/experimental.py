import nura
import nura.nn as nn
import nura.nn.utils as u
import nura.nn.functional as f
import numpy as np
import torch
import torch.nn as nn


def main():

    a = np.random.randn(64, 3, 5)
    b = a.swapaxes(1, 2)
    amu = a.mean(axis=(0, 1))
    bmu = b.mean(axis=(0, 2))
    cmu = a.reshape(64, -1, 3).mean(axis=(0, 2))
    print(a.shape, b.shape, amu, bmu, cmu, sep="\n")


if __name__ == "__main__":
    main()
