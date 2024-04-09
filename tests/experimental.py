import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    x = nura.randn(3, 4, 5, usegrad=True)
    p = f.softmax(x, dim=-1)
    loss = p.sum()
    loss.backward()


if __name__ == "__main__":
    main()
