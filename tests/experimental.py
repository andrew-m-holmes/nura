import numpy as np
import nura
import nura.nn as nn
from nura.autograd.function import Function


def main():

    a = nura.tensor(3.0, usegrad=True)
    b = nura.tensor(2.0)
    c = a * b
    d = a * c
    nura.backward(d)


if __name__ == "__main__":
    main()
