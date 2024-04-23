import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np
import gc


def main():

    x = nura.tensor(3.0, usegrad=True)
    x += 2
    y = x * 2
    print(y)
    y.backward()
    del y
    gc.collect()
    x += 2
    print(x.grad)


if __name__ == "__main__":
    main()
