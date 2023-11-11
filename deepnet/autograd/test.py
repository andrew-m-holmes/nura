from deepnet.autograd.functional import Mul, Add, Sub
from deepnet.tensor import Tensor


def main():
    a = Tensor([3], use_grad=True)
    b = Tensor([5], use_grad=True)

    mul = Add
    c = mul.apply(a, b)
    print(c)
    print(c.grad_fn.next_functions)
    c.backward(Tensor([1]))
    print(a.grad)
    print(b.grad)
    c.backward(Tensor([1]))
    print(a.grad)
    print(b.grad)


if __name__ == "__main__":
    main()
