import nura
import nura.nn as nn
import nura.autograd.functional as f
import numpy as np


def main():

    a = nura.randn(5, 1).int()
    a.dtype = nura.double
    print(a.version)
    print(a)


if __name__ == "__main__":
    main()
