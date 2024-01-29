import numpy as np
import deepnet
from deepnet.autograd.functional import grad

def main():

        a = deepnet.tensor(3., diff=True).float()
        b = deepnet.tensor(4., diff=True).float()
        c = deepnet.mul(a, b)
        e = deepnet.add(c, b)
        grads = grad((a, b, c), e, deepnet.tensor(1.))
        print(grads)

if __name__ == "__main__":
    main()
