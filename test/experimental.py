import nura
import nura.nn as nn
import nura.autograd.functional as f
import numpy as np


def main():

    a = np.random.rand(4, 3, 5)
    b = np.random.rand(1, 5, 2)
    c = np.dot(a, b)
    print(c.shape)


if __name__ == "__main__":
    main()
