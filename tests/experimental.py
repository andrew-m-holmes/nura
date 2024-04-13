import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np
import torch


def main():

    a = nura.rand(4, 3, 2, usegrad=True)
    b = nura.rand(2, usegrad=True)
    c = nura.dot(a, b)
    c.backward(nura.oneslike(c))


if __name__ == "__main__":
    main()
