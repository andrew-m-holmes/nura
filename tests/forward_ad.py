import deepnet
import deepnet.nn.functional as f


def main():

    a = deepnet.rand(use_grad=True)
    b = deepnet.rand()

    dual_a = deepnet.dual_tensor(a)
    dual_b = deepnet.dual_tensor(b)

    print(dual_a)
    print(dual_b)

    with deepnet.forward_ad():
        dual_c = dual_a + dual_b
    print(dual_c)
    c, tan_c = dual_c.unpack()
    print(c)
    c.backward()
    print(dual_a.primal.grad)


if __name__ == "__main__":
    main()
