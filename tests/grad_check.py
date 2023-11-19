import deepnet
import deepnet.nn.functional as f
import numpy as np


def main():

    a = deepnet.tensor(np.random.rand(1, 2, 3), use_grad=True)
    b = a.tranpose(-2, -1)
    print(b)
    b.backward()
    print(a.grad)


if __name__ == "__main__":
    main()
