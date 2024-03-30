import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn as torchnn
import torch.nn.functional as torchf


def main():

    x = nura.randn(3, 10).usedgrad()
    y = nura.randint(3, low=0, high=10)
    lossfn = nn.CrossEntropy()
    loss = lossfn(x, y)
    print(loss)
    loss.backward()
    print(x.grad)


if __name__ == "__main__":
    main()
