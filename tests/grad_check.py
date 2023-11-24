import deepnet
import deepnet.nn.functional as f
from deepnet.autograd.functional import vjp


def main():

    def func(x, w, b):
        return f.matmul(x, w) + b

    primals = (deepnet.tensor([[1., 2., 3.]]), deepnet.tensor(
        [[1., 1.], [2., 2.], [3., 3.]]), deepnet.tensor(.5))
    cotangent = deepnet.tensor([[1., 1.]])
    output, cotangents = vjp(primals, cotangent, func)
    print(output)
    print(cotangents)


if __name__ == "__main__":
    main()
