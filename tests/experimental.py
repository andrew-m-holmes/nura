import deepnet as dn
from deepnet.autograd.functional import grad, vjp, jvp


def main():
    
    a = dn.rand((64, 20), usegrad=True, dtype=dn.float)
    b = dn.cosine(a)
    print(b)


if __name__ == "__main__":
    main()
