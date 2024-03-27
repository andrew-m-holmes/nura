import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn.functional as tf


def main():

    a = np.random.randn(5)
    b = nura.tensor(a, usegrad=True).float()
    c = f.softmax(b)
    print(c)

    b = torch.from_numpy(b.data)
    c = tf.softmax(b)
    print(c)


if __name__ == "__main__":
    main()
