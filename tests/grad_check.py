import deepnet
import deepnet.nn.functional as f


def main():

    a = deepnet.tensor(1, use_grad=False)
    b = deepnet.tensor(2., use_grad=True)
    c = b * a
    print(c)
    c.backward()
    print(a.grad, b.grad)


if __name__ == "__main__":
    main()
