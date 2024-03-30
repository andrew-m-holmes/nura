import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn as torchnn
import torch.nn.functional as torchf


def main():

    padid = 0
    vocab_size = 3
    embeddingd_dim = 7
    batch_size = 1
    seq_len = 3

    x = nura.randint(batch_size, seq_len, low=0, high=vocab_size).long()
    embedding = nn.Embedding(vocab_size, embeddingd_dim, padid=padid)
    out = embedding(x)
    print(out)

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    embedding = torchnn.Embedding(vocab_size, embeddingd_dim, padding_idx=padid)
    out = embedding(x)
    print(out)


if __name__ == "__main__":
    main()
