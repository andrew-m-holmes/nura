import deepnet
import deepnet.nn.functional as f
from deepnet.autograd.forward_autograd import make_dual


def main():

    a = make_dual(deepnet.tensor(3.), deepnet.tensor(1.))
    b = make_dual(deepnet.tensor(5.), deepnet.tensor(1.))

    with deepnet.forward_autograd():
        c = f.mul(a, b)
        print(c)
        a = f.mul(c, a)
        print(a)


if __name__ == "__main__":
    main()
