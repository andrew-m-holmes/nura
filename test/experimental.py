import nura
import nura.nn as nn
import nura.autograd.functional as f
import numpy as np


def main():
    a = nura.randn(3, 5, usegrad=True)
    a.data = 2
    print(a)


if __name__ == "__main__":
    main()
