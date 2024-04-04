import nura
import nura.nn as nn
import nura.nn.functional as f


def main():

    x = nura.randn(3, 4, 5, usegrad=True)
    gamma = nura.ones(4, 5)
    beta = nura.zeros(4, 5)
    bias = False
    norm = f.layernorm(x, gamma, beta, dim=-1, bias=bias)
    print(norm)


if __name__ == "__main__":
    main()
