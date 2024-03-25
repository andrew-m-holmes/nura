import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f
import torch
import torch.nn.functional as tf


def main():
    w = nura.randn(4, 5, usegrad=True)
    x = nura.randint(2, 3, low=0, high=4).long()
    e = f.embedding(x, w, 0)
    print(e)
    l = e.sum()
    l.backward()
    print(w.grad)

    wdata, xdata = w.data, x.data
    w = torch.from_numpy(wdata).requires_grad_()
    x = torch.from_numpy(xdata)

    print()

    e = tf.embedding(x, w, 0)
    print(e)
    l = e.sum()
    l.backward()
    print(w.grad)


if __name__ == "__main__":
    main()
