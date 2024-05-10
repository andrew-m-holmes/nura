import nura
import nura.nn as nn


def main():

    a = nura.tensor(3.0, usegrad=True)
    b = nura.tensor(4.0, usegrad=True)
    c = a + b
    print(c)


if __name__ == "__main__":
    main()
