import nura
import nura.nn as nn
import nura.nn.utils as u
import nura.nn.functional as f
import numpy as np


def main():

    a = nura.randn(5, 1, 2, usegrad=True).double()
    b = a.mean(dim=(-2, -1), keepdims=True)
    c = a.std(correction=1, dim=(-3, -1), keepdims=True)
    c.backward()


if __name__ == "__main__":
    main()
