import deepnet
import deepnet.nn.functional as f
import deepnet.autograd.functional as af


def main():

    a = deepnet.tensor(5.)
    b = deepnet.tensor(7.)
    tan_a = deepnet.tensor(1.)
    tan_b = deepnet.tensor(1.)

    def func(a, b):
        return a * b

    output = af.jvp((a, b), (tan_a, tan_b), func)
    print(output)


if __name__ == "__main__":
    main()
