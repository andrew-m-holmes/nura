import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn as torchnn
import torch.nn.functional as torchf


def main():

    batch_size = 1
    seq_length = 5
    d_model = 10

    w0 = nura.randn(10, 10, usegrad=True)
    w1 = nura.randn(10, 10, usegrad=True)

    a = nura.randn(batch_size, 1, seq_length, d_model)
    b = nura.matmul(a, w0.T)  # batch_size, 1, seq_length, d_model
    c = b.reshape((-1, seq_length, d_model))  # (batch_size, seq_length, d_model)
    d = nura.matmul(c, w1.T)
    e = d.sum()  # scalar
    e.backward()


if __name__ == "__main__":
    main()
