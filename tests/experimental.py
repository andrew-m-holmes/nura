import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn as torchnn
import torch.nn.functional as torchf


def main():

    x = np.random.randn(3)
    y = np.random.choice(3, size=1)

    xn = nura.tensor(x).float()
    yn = nura.tensor(y).long()
    loss = f.crossentropy(xn, yn)
    print(loss)

    torch_x = torch.tensor(x).float()
    torch_y = torch.tensor(y).long()
    loss_torch = torchf.cross_entropy(torch_x, torch_y)
    print(loss_torch)


if __name__ == "__main__":
    main()
