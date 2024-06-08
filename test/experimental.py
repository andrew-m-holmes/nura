import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    x = nura.tensor(5).double()
    print(x)
    print(float(x))
    print(int(x))


if __name__ == "__main__":
    main()
