import numpy as np
import deepnet
from deepnet.autograd.functional import grad

def main():

    a = deepnet.rand(3, diff=True).float()
    b = deepnet.rand(3, diff=True).float()
    c = deepnet.mul(a, b)
    print(c)
    grads = grad((a, b), c, deepnet.ones_like(c))        
    print(grads)
    d = deepnet.tensor(1., diff=True).float()
    e = deepnet.tensor(2., diff=True).float()
    f = deepnet.div(d, e)
    print(f)

if __name__ == "__main__":
    main()
