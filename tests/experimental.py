import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn.functional as tf


def main():

    a = np.random.rand(2, 3).astype(np.float32)

    def softmax(z):
        e = np.exp(z - np.max(z))
        s = np.sum(e, axis=1, keepdims=True)
        return e / s

    def softmax_back(da, z):
        # z, da shapes - (m, n)
        m, n = z.shape
        p = softmax(z)
        # First we create for each example feature vector, it's outer product with itself
        # ( p1^2  p1*p2  p1*p3 .... )
        # ( p2*p1 p2^2   p2*p3 .... )
        # ( ...                     )
        tensor1 = np.einsum("ij,ik->ijk", p, p)  # (m, n, n)
        # Second we need to create an (n,n) identity of the feature vector
        # ( p1  0  0  ...  )
        # ( 0   p2 0  ...  )
        # ( ...            )
        tensor2 = np.einsum("ij,jk->ijk", p, np.eye(n, n))  # (m, n, n)
        # Then we need to subtract the first tensor from the second
        # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
        # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
        # ( ...                              )
        dSoftmax = tensor2 - tensor1
        # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
        dz = np.einsum("ijk,ik->ij", dSoftmax, da)  # (m, n)
        return dz

    grad = softmax_back(np.ones_like(a), a)
    print(grad)

    b = nura.tensor(a, usegrad=True).float()
    c = f.softmax(b, dim=-1)
    c.backward(nura.oneslike(c))
    print(b.grad)

    b = torch.from_numpy(b.data)
    b.requires_grad_()
    c = tf.softmax(b, dim=-1)
    c.backward(torch.ones_like(c))
    print(b.grad)


if __name__ == "__main__":
    main()
