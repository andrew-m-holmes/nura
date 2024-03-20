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

    bsize, seqlen, dm = 2, 4, 3

    q = nura.randn(bsize, seqlen, dm)
    k = q.clone()
    v = q.clone()

    mask = nura.tri(seqlen, seqlen).bool()
    ctx, attn = nn.selfattention(q, k, v, -1, mask)
    print(f"nura attn:\n{attn}")
    print(f"nura ctx:\n{ctx}")

    print("-" * 50)

    s = nura.matmul(q, k.transpose(-1, -2)) / dm**0.5
    s = nura.where(mask, s, -nura.inf)
    attn = nn.softmax(s, dim=-1)
    print(f"nura attn:\n{attn}")
    ctx = nura.matmul(attn, v)
    print(f"nura ctx:\n{ctx}")

    print("-" * 50)

    q = torch.from_numpy(q.data)
    k = q.clone()
    v = q.clone()

    s = torch.matmul(q, k.transpose(-1, -2)) / dm**0.5
    ones = torch.ones((seqlen, seqlen)).bool()
    mask = torch.tril(ones)
    s = s.masked_fill(mask == False, -1e9)
    attn = f.softmax(s, dim=-1)
    print(f"torch attn:\n{attn}")
    ctx = torch.matmul(attn, v)
    print(f"torch ctx:\n{ctx}")


if __name__ == "__main__":
    main()
