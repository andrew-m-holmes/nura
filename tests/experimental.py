import numpy as np
import deepnet


def main():

    a = deepnet.rand(3, diff=True).float()
    b = deepnet.rand(3, diff=True).float()
    c = a + b
    print(c)
    print(c.backfn)

if __name__ == "__main__":
    main()
