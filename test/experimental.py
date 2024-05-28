import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    x = nura.randint(0, 100, 7).float().attach()
    y = f.softmax(x, dim=-1)
    loss = y.sum()
    loss.backward()
    print(x.grad)


if __name__ == "__main__":
    main()
