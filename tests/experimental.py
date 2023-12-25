import torch
import torch.autograd.forward_ad as fwad
import torch.autograd.functional as f
import numpy as np
import deepnet

def main():

    a = torch.rand((1,)).float()
    b = torch.rand((1,)).float()
    v = torch.ones((1,)).float()

    output, jvp = f.jvp(torch.mul, (a, b), (v, v))
    print(output, jvp)
    
    with deepnet.forward_ad():
        a = deepnet.rand().float()
        b = deepnet.rand().float()
        a.dual(inplace=True)
        b.dual(inplace=True)
        c = deepnet.mul(a, b)
        print(c.tangent) 


if __name__ == "__main__":
    main()
