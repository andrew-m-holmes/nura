import deepnet
import deepnet.functional as f
from deepnet.autograd.functional import vjp, jvp
import numpy as np


def test_vjp_no_graph():
    a = np.random.rand(6, 4)
    b = np.random.rand(4, 10)
    c = np.random.rand(1)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    c_tensor = deepnet.tensor(c)
    primals = (a_tensor, b_tensor, c_tensor)
    cotangent = deepnet.ones((6, 10))

    def func(a, b, c):
        return f.add(f.matmul(a, b), c)

    result_tensor, cotangents = vjp(
        primals, cotangent, func, use_graph=False)
    expected = np.add(np.matmul(a, b), c)
    assert np.allclose(
        result_tensor.data,
        expected,
        rtol=1e-5,
        atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert primal.dim() == cotangent.dim()


def test_vjp_with_graph():
    a = np.random.rand(5, 3)
    b = np.random.rand(3, 7)
    c = np.random.rand(1)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    c_tensor = deepnet.tensor(c, use_grad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    cotangent = deepnet.ones((5, 7))

    def func(a, b, c):
        return f.add(f.matmul(a, b), c)

    result_tensor, cotangents = vjp(
        primals, cotangent, func, use_graph=True)
    result_tensor.backward(cotangent)
    expected = np.add(np.matmul(a, b), c)
    np.testing.assert_allclose(
        result_tensor.data,
        expected,
        rtol=1e-5,
        atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert primal.dim() == cotangent.dim()
        assert np.allclose(
            primal.grad.data,
            cotangent.data,
            rtol=1e-5,
            atol=1e-5)


def test_vjp_add_sub():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    c = np.random.rand(4, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    c_tensor = deepnet.tensor(c, use_grad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    cotangent = deepnet.ones((4, 4))

    def func(a, b, c):
        return f.sub(f.add(a, b), c)

    result_tensor, cotangents = vjp(
        primals, cotangent, func, use_graph=True)
    result_tensor.backward(cotangent)
    expected = np.subtract(np.add(a, b), c)
    np.testing.assert_allclose(
        result_tensor.data,
        expected,
        rtol=1e-5,
        atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(
            primal.grad.data,
            cotangent.data,
            rtol=1e-5,
            atol=1e-5)


def test_vjp_mul_div():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)
    c = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    c_tensor = deepnet.tensor(c, use_grad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    cotangent = deepnet.ones((5, 5))

    def func(a, b, c):
        return f.div(f.mul(a, b), c)

    result_tensor, cotangents = vjp(
        primals, cotangent, func, use_graph=True)
    result_tensor.backward(cotangent)
    expected = np.divide(np.multiply(a, b), c)
    np.testing.assert_allclose(
        result_tensor.data,
        expected,
        rtol=1e-5,
        atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(
            primal.grad.data,
            cotangent.data,
            rtol=1e-5,
            atol=1e-5)


def test_vjp_matmul_sum():
    a = np.random.rand(3, 4)
    b = np.random.rand(4, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    primals = (a_tensor, b_tensor)
    cotangent = deepnet.ones((3,))

    def func(a, b):
        return f.sum(f.matmul(a, b), dims=0)

    result_tensor, cotangents = vjp(
        primals, cotangent, func, use_graph=True)
    result_tensor.backward(cotangent)
    expected = np.sum(np.matmul(a, b), axis=0)
    np.testing.assert_allclose(
        result_tensor.data,
        expected,
        rtol=1e-5,
        atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(
            primal.grad.data,
            cotangent.data,
            rtol=1e-5,
            atol=1e-5)


def test_vjp_add_mul_broadcast():
    a = np.random.rand(3, 4)
    b = np.random.rand(4,)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    primals = (a_tensor, b_tensor)
    cotangent = deepnet.ones((3, 4))

    def func(a, b):
        return f.mul(f.add(a, b), b)

    result_tensor, cotangents = vjp(
        primals, cotangent, func, use_graph=True)
    result_tensor.backward(cotangent)
    expected = np.multiply(np.add(a, b), b)
    np.testing.assert_allclose(
        result_tensor.data,
        expected,
        rtol=1e-5,
        atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(
            primal.grad.data,
            cotangent.data,
            rtol=1e-5,
            atol=1e-5)


def test_vjp_nested_operations_broadcast():
    a = np.random.rand(5, 6)
    b = np.random.rand(6,)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    primals = (a_tensor, b_tensor)
    cotangent = deepnet.ones((5, 6))

    def func(a, b):
        return f.cosine(f.div(f.add(a, b), f.sine(b)))

    result_tensor, cotangents = vjp(
        primals, cotangent, func, use_graph=True)
    result_tensor.backward(cotangent)
    expected = np.cos(np.divide(np.add(a, b), np.sin(b)))
    np.testing.assert_allclose(
        result_tensor.data,
        expected,
        rtol=1e-5,
        atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(
            primal.grad.data,
            cotangent.data,
            rtol=1e-5,
            atol=1e-5)


def test_vjp_matmul_add_broadcast():
    a = np.random.rand(4, 3)
    b = np.random.rand(3, 5)
    c = np.random.rand(5,)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    c_tensor = deepnet.tensor(c, use_grad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    cotangent = deepnet.ones((4, 5))

    def func(a, b, c):
        return f.add(f.matmul(a, b), c)

    result_tensor, cotangents = vjp(
        primals, cotangent, func, use_graph=True)
    result_tensor.backward(cotangent)
    expected = np.add(np.matmul(a, b), c)
    np.testing.assert_allclose(
        result_tensor.data,
        expected,
        rtol=1e-5,
        atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(
            primal.grad.data,
            cotangent.data,
            rtol=1e-5,
            atol=1e-5)


def main():

    # Basic VJP Tests

    test_vjp_no_graph()
    test_vjp_with_graph()

    # Composition Function VJP Tests

    test_vjp_add_sub()
    test_vjp_mul_div()
    test_vjp_matmul_sum()

    # Broadcasting Function VJP Tests

    test_vjp_add_mul_broadcast()
    test_vjp_nested_operations_broadcast()
    test_vjp_matmul_add_broadcast()

    print("All tests passed")


if __name__ == "__main__":
    main()
