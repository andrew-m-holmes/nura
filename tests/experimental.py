import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn.functional as torchf


def main():
    x = np.random.randn(3, 5).astype(np.float32)
    y = np.random.randint(0, 5, (3,))

    nx = nura.tensor(x, usegrad=True)
    ny = nura.tensor(y)
    loss = f.crossentropy(nx, ny, ignoreid=0)
    print(loss)

    tx = torch.from_numpy(x)
    ty = torch.from_numpy(y)
    loss = torchf.cross_entropy(tx, ty, ignore_index=0)
    print(loss)


if __name__ == "__main__":
    main()
