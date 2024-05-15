import nura
import nura.nn as nn
import nura.autograd.functional as f
import numpy as np


def main():

    a = np.random.rand(4, 3, 5)
    b = np.random.rand(3, 2, 5)
    c = np.inner(a, b)
    print(c.shape)


if __name__ == "__main__":
    main()
