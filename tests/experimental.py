import numpy as np
import nura
import nura.nn as nn
from nura.autograd.graph import toposort


def main():

    a = nura.tensor([1.0, 2.0, 3], usegrad=True)
    b = a + 2
    c = a * b
    d = a * c
    assert d.gradfn is not None
    topolist = toposort(d.gradfn)
    print(topolist)


if __name__ == "__main__":
    main()
