import torch
import torch.autograd.functional as tf
import deepnet.autograd.functional as df
import numpy as np
import deepnet

def main():

    a = torch.rand((1,)).float()
    b = torch.rand((1,)).float()
    v = torch.ones((1,)).float()

    output, jvp = tf.jvp(torch.mul, (a, b), (v, v))
    print(output, jvp)
    
    a = deepnet.tensor(a.numpy())
    b = deepnet.tensor(b.numpy())
    v = deepnet.ones((1,))

    output, jvp = df.jvp((a, b), (v, v), deepnet.mul)
    print(output, jvp)

if __name__ == "__main__":
    main()
