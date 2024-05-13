import nura
import nura.nn as nn
from nura.autograd.forwardad import primal


def main():

    a = nura.tensor(3.0, usegrad=True)
    print(a)
    with nura.forwardmode():
        print(a)
        b = nura.forwardad.primal(a, 30.0)
        c = b * 10
        print(c.grad)
    print(c.grad)


if __name__ == "__main__":
    main()
