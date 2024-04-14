import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    a = nura.rand(5)
    b = nura.rand(5)
    c = a % b
    print(c)


if __name__ == "__main__":
    main()
