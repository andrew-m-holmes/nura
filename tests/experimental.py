import numpy as np
import deepnet
from deepnet.autograd.functional import grad, backward

def main():

        a = deepnet.tensor(3., diff=True).float()
        b = deepnet.tensor(4., diff=True).float()
        c = deepnet.mul(a, b)
        grads = grad((a, b, c), c, deepnet.tensor(1.))
        print(grads)
        backward(c)
        print(a.grad, b.grad)
        

if __name__ == "__main__":
    main()
