import nura
import nura.nn as nn
from nura.nn.modules.multihead import MultiHeadAttention


def main():

    m, seqlen, dm = 3, 10, 64
    dk = dv = 16
    heads = 4
    mha = MultiHeadAttention(dm, dk, dv, heads, dim=-1)
    q = nura.randn(m, seqlen - 1, dm) * 0.5
    k = nura.randn(m, seqlen, dm) * 0.5
    v = nura.randn(m, seqlen, dm) * 0.5
    mask = nura.tri(seqlen - 1, seqlen).unsqueeze(0)

    ctx, attn = mha(q, k, v, mask=mask)
    print(mha)


if __name__ == "__main__":
    main()
