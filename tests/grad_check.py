import deepnet
import deepnet.nn.functional as f


def main():

    with deepnet.forward_autograd():
        a = deepnet.tensor(5., use_grad=True)
        b = deepnet.tensor(6., use_grad=True)
        c = a * b
        print(c.tangent)


if __name__ == "__main__":
    main()
