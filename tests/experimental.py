import nura
import nura.nn as nn
from nura.nn.modules.multihead import MultiHeadAttention


def main():

    n = 5
    l = 3
    b = 2
    i = nura.randint(b, l, low=0, high=n)
    print(i)
    o = nura.onehot(i, n)
    print(o)


if __name__ == "__main__":
    main()
