import nura
import nura.nn as nn
import nura.autograd.functional as f


def main():

    def func(a, b, c):
        return a * b + c

    a = nura.randn(3)
    b = nura.randn(3)
    c = nura.randn(1)

    output, jacobian = f.jacfwd((a, b, c), func, 0)
    print(output)
    print(jacobian)


if __name__ == "__main__":
    main()
