import deepnet
import deepnet.nn.functional as f
from deepnet.autograd.forward_autograd import make_dual


def main():

    a = deepnet.tensor(3., use_grad=True)
    b = deepnet.tensor(4., use_grad=True)
    a = make_dual(a, deepnet.tensor(1.))
    b = make_dual(b, deepnet.tensor(1.))
    with deepnet.forward_autograd():
        c = f.mul(a, b)
        d = f.mul(c, a)
        print(d)


if __name__ == "__main__":
    main()
