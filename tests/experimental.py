import numpy as np
import nura
import nura.nn as nn
from nura.nn.modules.multihead import MultiHeadAttention
from nura.nn.modules.embedding import Embedding


def main():

    batch_size = 1
    seq_len = 5
    d_model = 64
    d_k = 16
    d_v = 16
    heads = 4
    vocab_size = 10
    x = nura.randint(batch_size, seq_len, low=0, high=vocab_size).float()
    embedding = Embedding(d_model, vocab_size)
    multihead = MultiHeadAttention(d_model, d_k, d_v, heads)

    x = embedding(x) * 0.25
    ctx, attn = multihead(x, x, x)
    print(ctx.dim, attn.dim)
    ctx.backward(nura.oneslike(ctx))


if __name__ == "__main__":
    main()
