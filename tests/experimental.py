import numpy as np
import nura
import nura.nn as nn
from nura.nn.modules.multihead import MultiHeadAttention
from nura.nn.modules.embedding import Embedding


def main():

    a = nura.randint(1, 5, low=-10, high=11)
    print(a)
    i = nura.indexwhere(a > 0)
    print(np.where(a.data > 0))
    print(i)


if __name__ == "__main__":
    main()
