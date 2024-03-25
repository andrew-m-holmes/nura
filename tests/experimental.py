import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

# import torch
# import torch.nn.functional as tf


def main():

    a = nura.randint(3, 3, low=0, high=2).float().usedgrad()
    print(a + 1)


if __name__ == "__main__":
    main()
