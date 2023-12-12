import torch
import numpy as np
import deepnet
import deepnet.functional as f


def main():

    a = deepnet.tensor([3., 4, 2], use_grad=True)
    dual_a = deepnet.dual_tensor(a)
    print(dual_a)


if __name__ == "__main__":
    main()
