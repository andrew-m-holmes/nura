import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np
import torch


def main():
    x = np.random.randn(3, 4)
    z = nura.tensor(x, usegrad=True, dtype=nura.float)
    a = f.softmax(z)
    a.backward(nura.oneslike(a))
    print(z.grad)

    z = torch.from_numpy(x).float().requires_grad_()
    a = torch.nn.functional.softmax(z, dim=-1)
    a.backward(torch.ones_like(a))
    print(z.grad)


if __name__ == "__main__":
    main()
