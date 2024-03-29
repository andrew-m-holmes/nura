import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn.functional as torchf


def main():

    ignoreid = np.random.choice(5)

    x = np.random.randn(3, 5).astype(np.float32)
    y = np.random.randint(0, 5, (3,)).astype(np.int32)

    nx = nura.tensor(x, usegrad=True)
    ny = nura.tensor(y)
    loss = f.crossentropy(nx, ny, ignoreid=ignoreid)
    print(loss)
    loss.backward()
    print(nx.grad)

    tx = torch.from_numpy(x).requires_grad_()
    ty = torch.from_numpy(y).long()
    loss = torchf.cross_entropy(tx, ty, ignore_index=ignoreid)
    print(loss)
    loss.backward()
    print(tx.grad)


if __name__ == "__main__":
    main()
