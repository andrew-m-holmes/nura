import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    def rmse(a, y):
        return np.sqrt(np.mean(np.square(a - y)))

    a, y = np.random.rand(4), np.random.randint(0, 2, size=4).astype(float)
    error = rmse(a, y)
    print(error)

    a = nura.tensor(a)
    y = nura.tensor(y)
    error = f.rmse(a, y)
    print(error)


if __name__ == "__main__":
    main()
