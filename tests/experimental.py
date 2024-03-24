import nura
import nura.nn as nn
from nura.nn.modules.multihead import MultiHeadAttention
from nura.nn.modules.embedding import Embedding


def main():

    batch_size = 1
    seq_len = 7
    vocab_size = 5
    embed_dim = 4

    x = nura.randint(batch_size, seq_len, low=0, high=vocab_size).float()
    embed = Embedding(embed_dim, vocab_size)
    out = embed(x)
    print(out.dim)


if __name__ == "__main__":
    main()
