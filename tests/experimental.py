import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn as torchnn
import torch.nn.functional as torchf


def main():

    w = np.arange(start=1, stop=5).reshape(2, 2).astype(np.float32)
    x = np.array([0, 1, 1])[None]
    e = w[x]
    g = np.array([1.0, 2.0, 3.0]).reshape(1, 3, 1) + np.zeros_like(e)

    w = nura.tensor(w, usegrad=True)
    x = nura.tensor(x)
    e = f.embedding(x, w, padid=1)
    g = nura.tensor(g)
    e.backward(g)

    print(f"w:\n{w}\n")
    print(f"x:\n{x}\n")
    print(f"e:\n{e}\n")
    print(f"g:\n{g}\n")
    print(f"w.grad:\n{w.grad}")


if __name__ == "__main__":
    main()
