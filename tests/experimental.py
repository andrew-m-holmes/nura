import numpy as np
import deepnet
from deepnet.autograd.functional import grad

def main():

    a = deepnet.rand(3, mut=True).float()
    b = deepnet.rand(3, mut=True).float()
    c = deepnet.mul(a, b)
    deepnet.backward(c, deepnet.oneslike(c))
    print(a.grad)
    print(b.grad)

if __name__ == "__main__":
    main()
