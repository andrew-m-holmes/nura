import deepnet
import deepnet.nn.functional as f


def test_add():
    pass


def test_mul():
    pass


def main():

    a = deepnet.tensor(3, use_grad=True)
    b = deepnet.tensor(4, use_grad=True)
    c = a * b

    print(c)
    c.backward()
    print(a.grad, b.grad)
    print(deepnet.tensor(1, use_grad=1))

    assert a.grad.data == b.data
    assert b.grad.data == a.data


if __name__ == "__main__":
    main()
