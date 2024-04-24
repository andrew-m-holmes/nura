import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    x = nura.randn(5, 3, 2).attached()
    y = nura.randn(3, 1).attached()
    z = x + y
    z.backward(nura.oneslike(z))


if __name__ == "__main__":
    main()
