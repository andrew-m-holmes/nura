import deepnet
import deepnet.nn.functional as f
from deepnet.autograd.forward_autograd import make_dual


def main():

    a = deepnet.randn((3, 4, 2, 1, 3, 1))
    a.use_grad = True
    print(a.dim())
    b = a.squeeze()
    print(b.dim())
    print(b.grad_fn)
    b.backward()
    print(a.grad.dim())


if __name__ == "__main__":
    main()
