import numpy as np
import deepnet as dn
from deepnet.autograd.functional import grad

def main():

    a = dn.tensor(5., usegrad=True).float()
    b = dn.tensor(3., usegrad=True).float()
    c = dn.sub(a, b)
    d = dn.mul(c, a)
    d.backward()
    print(a.grad)
    print(b.grad)
    print(c.grad)

if __name__ == "__main__":
    main()
