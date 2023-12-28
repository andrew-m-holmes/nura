import torch
import torch.autograd.functional as taf
import deepnet.autograd.functional as daf
import numpy as np
import deepnet


def main():

    a = torch.rand((1,)).float()
    b = torch.rand((1,)).float()

    v = torch.ones_like(a)
    output, jvp = taf.jvp(torch.div, (a, b), (v, v))
    print(output, jvp, sep="\n")
    
    a = deepnet.tensor(a.numpy())
    b = deepnet.tensor(b.numpy())
    v = deepnet.ones_like(a)
    output, jvp = daf.jvp((a, b), (v, v), deepnet.div)
    print(output, jvp, sep="\n")

if __name__ == "__main__":
    main()
