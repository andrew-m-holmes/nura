import deepnet
import deepnet.nn.functional as f


def main():
    a = deepnet.randn((4, 1), use_grad=True, dtype=deepnet.float)
    b = deepnet.ones((4, 1), use_grad=True, dtype=deepnet.float)
    c = f.mul(a, b)

    c.backward()
    print(a.grad)
    print(b.grad)


if __name__ == "__main__":
    main()
