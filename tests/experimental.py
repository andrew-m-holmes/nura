import numpy as np
import deepnet
import deepnet.functional as f
from deepnet.autograd.functional import vjp, jvp

def main():
    
    a = np.random.rand(4, 5, 6)
    b = a.sum()
    c = np.random.rand(1)
    d = np.broadcast_to(b, (1,))
    print(d.shape)

if __name__ == "__main__":
    main()
