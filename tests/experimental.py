import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn.functional as tf


def main():

    a = np.random.randn(5, 4)
    b = nura.tensor(a, usegrad=True).float()
    c = f.softmax(b)
    print(c)
    c.backward(nura.oneslike(c))
    print(b.grad)

    b = torch.from_numpy(b.data)
    b.requires_grad_()
    c = tf.softmax(b, dim=-1)
    print(c)
    c.backward(torch.ones_like(c))
    print(b.grad)


if __name__ == "__main__":
    main()
