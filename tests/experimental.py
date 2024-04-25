import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    a = np.random.randint(0, 10, (2, 3, 4))
    x = nura.tensor(a)
    for i in x[0, 1, 0:3]:
        print(i)
    for i in a[0, 1, 0:3]:
        print(i)

    iterx = iter(x)
    for i in iterx:
        print(i)

    itera = iter(a)
    for i in itera:
        print(i)


if __name__ == "__main__":
    main()
