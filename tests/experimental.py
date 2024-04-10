import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    x = nura.rand(3, 4).half()
    y = f.dropout(x, 0.9)
    print(1 / 1e-9)
    pass


if __name__ == "__main__":
    main()
