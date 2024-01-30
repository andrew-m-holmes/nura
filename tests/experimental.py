import numpy as np
import deepnet
from deepnet.autograd.functional import grad, backward

def main():

        a = deepnet.tensor(3., diff=True).float()
        b = deepnet.tensor(4., diff=True).float()
        c = deepnet.mul(a, b)
        

if __name__ == "__main__":
    main()
