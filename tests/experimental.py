import nura
import nura.nn as nn
import nura.autograd.functional as f


def main():

    x = nura.randn(5, 4)
    w = nura.randn(6, 4, usegrad=True)
    b = nura.randn(6, usegrad=True)

    def func(x, w, b):
        return (x @ w.T + b).sum()

    output, vjps = f.vjp((x, w, b), nura.rand(), func)


if __name__ == "__main__":
    main()
