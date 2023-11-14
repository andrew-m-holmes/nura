import deepnet.nn.functional as f
from deepnet.tensor import tensor


def test_add():
    pass


def test_mul():
    pass


def main():

    a = tensor(3, use_grad=True)
    b = tensor(4, use_grad=True)
    c = a * b

    print(c)
    c.backward()
    print(a.grad, b.grad)
    print(tensor(1, use_grad=1))

    assert a.grad == b.data
    assert b.grad == a.data


if __name__ == "__main__":
    main()
