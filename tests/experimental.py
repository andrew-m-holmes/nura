import nura
import nura.nn as nn
import torch
import torch.nn.functional as f
import numpy as np


def main():

    class Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 8)

        def forward(self, x):
            return self.linear(x)

    bsize, seqlen, dm = 2, 5, 3

    wq = nura.randn(dm, dm)
    wk = nura.randn(dm, dm)
    wv = nura.randn(dm, dm)

    q = nura.randn(bsize, seqlen, dm)

    q = nn.linear(q, wq)
    k = nn.linear(q, wk)
    v = nn.linear(q, wv)

    mask = nura.tri(seqlen, seqlen).bool()
    ctx, attn = nn.selfattention(q, k, v, mask)
    print(ctx, attn, sep="\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
