import nura
import nura.nn as nn
import nura.nn.utils as u
import nura.nn.functional as f
import numpy as np


def main():

    x = np.random.randn(3, 4)
    print(x)
    x = x.swapaxes(1, -1)
    print(x)
    print(x.shape)


if __name__ == "__main__":
    main()
