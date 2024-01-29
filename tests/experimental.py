import numpy as np
import deepnet


def main():

    a = deepnet.rand(3, diff=True).float()
    b = deepnet.rand(3, diff=True).float()
    c = deepnet.add(a, b)
    c.backward(deepnet.ones_like(c))
    print(a.grad)
    print(b.grad)

if __name__ == "__main__":
    main()
