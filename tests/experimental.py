import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn as torchnn
import torch.nn.functional as torchf


def main():

    p = 0.5
    x = nura.randn(3, 5, usegrad=True)
    print(x.dtype)
    dropout = nn.Dropout(p)
    dropout.train()
    z = dropout(x)
    loss = z.sum()
    loss.backward()
    print(dropout)
    print(z)
    print(x.grad)


if __name__ == "__main__":
    main()
