import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    x = np.random.randn(7)
    exp = np.exp(x - x.max(keepdims=True))
    a = np.log(exp * (1 / exp.sum(keepdims=True)))
    print(a)
    print(np.exp(a).sum())

    xmax = x.max(keepdims=True)
    a_ = x - xmax - np.log(np.exp(x - xmax).sum(keepdims=True))
    print(a_)
    print(np.exp(a_).sum())

    np.testing.assert_allclose(a_, a, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    main()
