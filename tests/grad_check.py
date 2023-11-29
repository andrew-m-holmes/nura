import deepnet
import deepnet.nn.functional as f


def main():

    with deepnet.forward_autograd():
        a = deepnet.tensor(5.)
        b = deepnet.tensor(7.)
        a = deepnet.dual_tensor(a)
        b = deepnet.dual_tensor(b)
        print(a)
        print(b)
        c = a * b
        print(c)


if __name__ == "__main__":
    main()
