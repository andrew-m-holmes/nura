import numpy as np
import deepnet as dn
from deepnet.autograd.functional import grad


def main():

    a = dn.tensor(5.0, usegrad=True).float()
    b = dn.tensor(3.0, usegrad=True).float()

    c = a * 3. + b
    c.backward()
    print(a.grad)
    print(b.grad)

if __name__ == "__main__":
    main()
