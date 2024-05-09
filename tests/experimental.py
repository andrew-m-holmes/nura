import numpy as np
import nura
import nura.nn as nn


def main():

    a = nura.tensor(3.0, usegrad=True)
    b = nura.tensor(4.0, usegrad=True)
    c = a * b
    c += 1
    d = a - c
    c += 1
    d.backward()
    e = d * 15.0
    e.backward(input=c)
    print(c.grad)


if __name__ == "__main__":
    main()
