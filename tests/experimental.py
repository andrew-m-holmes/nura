import nura
import nura.nn as nn


def main():

    a = nura.tensor(3.0, usegrad=True)
    with nura.forwardmode():
        b = nura.forwardad.primal(a, 10.0)
        c = nura.forwardad.primal(20.0, 8.0)
        print(b.grad)
        d = b * c
        print(d.grad)


if __name__ == "__main__":
    main()
