import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np
import torch


def main():

    x = nura.randn(3, 5, usegrad=True)
    o = f.sigmoid(x)
    y = nura.randint(3, low=0, high=5, dtype=nura.int)
    loss = f.crossentropy(o, y, ignoreid=0, reduction="sum")
    loss.backward()


if __name__ == "__main__":
    main()
