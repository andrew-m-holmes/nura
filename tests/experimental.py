import numpy as np
import nura
import nura.nn as nn


def main():

    a = nura.tensor([1.0, 2.0, 3], usegrad=True)
    b = a + 2
    print(b)


if __name__ == "__main__":
    main()
