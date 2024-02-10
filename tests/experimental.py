import deepnet as dn
from deepnet.autograd.functional import grad, vjp, jvp


def main():
    
    a = dn.rand((4, 1, 4, 1), usegrad=True, dtype=dn.float)
    b = a.squeeze()
    b.backward(dn.oneslike(b))
    print(a.grad.dim)


if __name__ == "__main__":
    main()
