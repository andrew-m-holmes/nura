import deepnet
import deepnet.nn.functional as f
import deepnet.autograd.functional as af


def main():

    a = deepnet.tensor(5., dtype=deepnet.double)
    b = deepnet.tensor(7.)
    print(a)
    print(b)


if __name__ == "__main__":
    main()
