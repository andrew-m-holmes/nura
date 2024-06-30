import nura
import nura.nn as nn
import nura.nn.utils as u
import nura.nn.functional as f
import numpy as np


def main():

    a = nura.randn(3, 3, 4, 5, usegrad=True)
    b = nura.flatten(a, 1, 2)
    print(b.dim)
    b.backward(nura.oneslike(b))


if __name__ == "__main__":
    main()
