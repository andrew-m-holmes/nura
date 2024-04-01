import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn as torchnn
import torch.nn.functional as torchf

np._set_promotion_state("weak_and_warn")


def main():

    a = np.random.rand(3, 4).astype(np.float32)
    b = a * np.power(a, a - 1)


if __name__ == "__main__":
    main()
