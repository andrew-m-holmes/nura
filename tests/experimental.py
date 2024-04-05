import nura
import nura.nn as nn
import nura.nn.functional as f


def main():

    x = nura.randn(3, 4, usegrad=True)
    gamma = nura.ones(4)
    beta = nura.zeros(4)
    bias = False
    norm = f.layernorm(x, gamma, beta, dim=-1, bias=bias)
    print(norm)
    norm.backward(nura.oneslike(norm))
    print(x.grad)


if __name__ == "__main__":
    main()
