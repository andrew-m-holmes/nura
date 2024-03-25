import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

# import torch
# import torch.nn.functional as tf


def main():

    embedding = nn.Embedding(64, 100, padid=0)
    multihead = nn.MultiHeadAttention(64, 16, 16, 4)


if __name__ == "__main__":
    main()
