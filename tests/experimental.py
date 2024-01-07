import numpy as np
import torch
import torch.autograd as torch_autograd
import deepnet
import deepnet.autograd as deepnet_autograd


def main():
    a = torch.tensor(3., requires_grad=True)
    b = torch.tensor(8., requires_grad=True)
    output = torch.div(a, b)
    df_da, df_db = torch_autograd.grad(output, (a, b))
    print(df_da, df_db)


if __name__ == "__main__":
    main()
