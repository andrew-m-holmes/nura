import numpy as np
import torch
import torch.autograd as torch_autograd
import deepnet
import deepnet.autograd as deepnet_autograd
from deepnet.autograd.functional import vjp, jvp, grad, jacobian


def main():
    a = torch.tensor(3., requires_grad=True)
    b = torch.tensor(8., requires_grad=True)
    output = torch.sub(torch.mul(torch.div(a, b), a), b)
    df_da, df_db = torch_autograd.grad(output, (a, b))
    print(df_da, df_db)

    a = deepnet.tensor(3., use_grad=True).double()
    b = deepnet.tensor(8., use_grad=True)
    output = deepnet.sub(deepnet.mul(deepnet.div(a, b), a), b)
    df_da, df_db = grad((a, b), output)
    print(df_da, df_db)

if __name__ == "__main__":
    main()
