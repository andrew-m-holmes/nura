import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np
import torch


def main():

    x = nura.randn(3, 5, usegrad=True)
    w = nura.randn(5, 1)
    o = nura.dot(x, w)
    print(o)


if __name__ == "__main__":
    main()
