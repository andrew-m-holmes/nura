import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    def mse(a, y):
        return np.mean(0.5 * np.square(a - y))

    a, y = np.random.rand(4), np.random.randint(0, 2, size=4).astype(float)
    error = mse(a, y)
    print(error)

    a = nura.tensor(a)
    y = nura.tensor(y)
    error = f.mse(a, y)
    print(error)

    print(nura.sum(nura.square(a - y)) / a.nelem)


if __name__ == "__main__":
    main()
