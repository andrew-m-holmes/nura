import deepnet
import deepnet.functional as f
from deepnet.autograd.functional import vjp, jvp, grad
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
    assert np.allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
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
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert primal.dim() == cotangent.dim()
        assert np.allclose(primal.grad.data, cotangent.data,
                           rtol=1e-5, atol=1e-5)


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
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(primal.grad.data, cotangent.data,
                           rtol=1e-5, atol=1e-5)


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
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(primal.grad.data, cotangent.data,
                           rtol=1e-5, atol=1e-5)


def test_vjp_matmul_sum():
    a = np.random.rand(3, 4)
    b = np.random.rand(4, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    primals = (a_tensor, b_tensor)
    cotangent = deepnet.ones((3,))

    def func(a, b):
        return f.sum(f.matmul(a, b), dims=0)

    result_tensor, cotangents = vjp(primals, cotangent, func, use_graph=True)
    result_tensor.backward(cotangent)
    expected = np.sum(np.matmul(a, b), axis=0)
    np.testing.assert_allclose(
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(primal.grad.data, cotangent.data,
                           rtol=1e-5, atol=1e-5)


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
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(primal.grad.data, cotangent.data,
                           rtol=1e-5, atol=1e-5)


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
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(primal.grad.data, cotangent.data,
                           rtol=1e-5, atol=1e-5)


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
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert np.allclose(primal.grad.data, cotangent.data,
                           rtol=1e-5, atol=1e-5)


def test_jvp_no_graph():
    a = np.random.rand(6, 4)
    b = np.random.rand(10, 4)
    c = np.random.rand(1)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    c_tensor = deepnet.tensor(c)
    primals = (a_tensor, b_tensor, c_tensor)
    tangents = (deepnet.ones_like(a_tensor), deepnet.ones_like(
        b_tensor), deepnet.ones_like(c_tensor))

    def func(a, b, c):
        return f.add(f.matmul(a, b.transpose()), c)

    result_tensor, output_tangent = jvp(
        primals, tangents, func, use_graph=False)
    expected = np.add(np.matmul(a, b.T), c)
    assert np.allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim() == output_tangent.dim()


def test_jvp_with_graph():
    a = np.random.rand(5, 3)
    b = np.random.rand(3, 7)
    c = np.random.rand(1)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    c_tensor = deepnet.tensor(c, use_grad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    tangents = (deepnet.ones_like(a_tensor), deepnet.ones_like(
        b_tensor), deepnet.ones_like(c_tensor))

    def func(a, b, c):
        return f.add(f.matmul(a, b), c)

    result_tensor, output_tangent = jvp(primals, tangents, func, use_graph=True)
    result_tensor.backward(output_tangent)
    expected = np.add(np.matmul(a, b), c)
    np.testing.assert_allclose(
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim() == output_tangent.dim()
    for primal in primals:
        assert primal.grad is not None


def test_jvp_add_sub():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    c = np.random.rand(4, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    c_tensor = deepnet.tensor(c, use_grad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    tangents = (deepnet.ones_like(a_tensor), deepnet.ones_like(
        b_tensor), deepnet.ones_like(c_tensor))

    def func(a, b, c):
        return f.sub(f.add(a, b), c)

    result_tensor, output_tangent = jvp(primals, tangents, func, use_graph=True)
    result_tensor.backward(output_tangent)
    expected = np.subtract(np.add(a, b), c)
    np.testing.assert_allclose(
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim() == output_tangent.dim()
    for primal in primals:
        assert primal.grad is not None


def test_jvp_mul_div():
    a = np.random.rand(7, 7)
    b = np.random.rand(7, 7)
    c = np.random.rand(7, 7)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    c_tensor = deepnet.tensor(c, use_grad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    tangents = (deepnet.ones_like(a_tensor), deepnet.ones_like(
        b_tensor), deepnet.ones_like(c_tensor))

    def func(a, b, c):
        return f.div(f.mul(a, b), c)

    result_tensor, output_tangent = jvp(primals, tangents, func, use_graph=True)
    result_tensor.backward(output_tangent)
    expected = np.divide(np.multiply(a, b), c)
    np.testing.assert_allclose(
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim() == output_tangent.dim()
    for primal in primals:
        assert primal.grad is not None


def test_jvp_matmul_sum():
    a = np.random.rand(3, 8)
    b = np.random.rand(8, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    primals = (a_tensor, b_tensor)
    tangents = (deepnet.ones_like(a_tensor), deepnet.ones_like(b_tensor))

    def func(a, b):
        return f.sum(f.matmul(a, b), dims=0)

    result_tensor, output_tangent = jvp(primals, tangents, func, use_graph=True)
    result_tensor.backward(output_tangent)
    expected = np.sum(np.matmul(a, b), axis=0)
    np.testing.assert_allclose(
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim() == output_tangent.dim()
    for primal in primals:
        assert primal.grad is not None


def test_jvp_add_mul_broadcast():
    a = np.random.rand(3, 4)
    b = np.random.rand(4,)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    primals = (a_tensor, b_tensor)
    tangents = (deepnet.ones_like(a_tensor), deepnet.ones_like(b_tensor))

    def func(a, b):
        return f.mul(f.add(a, b), b)

    result_tensor, output_tangent = jvp(primals, tangents, func, use_graph=True)
    result_tensor.backward(output_tangent)
    expected = np.multiply(np.add(a, b), b)
    np.testing.assert_allclose(
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim() == output_tangent.dim()
    for primal in primals:
        assert primal.grad is not None


def test_jvp_nested_operations_broadcast():
    a = np.random.rand(5, 4)
    b = np.random.rand(4,)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    primals = (a_tensor, b_tensor)
    tangents = (deepnet.ones_like(a_tensor), deepnet.ones_like(b_tensor))

    def func(a, b):
        return f.cosine(f.div(f.add(a, b), f.sine(b)))

    result_tensor, output_tangent = jvp(primals, tangents, func, use_graph=True)
    result_tensor.backward(output_tangent)
    expected = np.cos(np.divide(np.add(a, b), np.sin(b)))
    np.testing.assert_allclose(
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim() == output_tangent.dim()
    for primal in primals:
        assert primal.grad is not None


def test_jvp_sine_permute_sum_broadcast():
    a = np.random.rand(4, 5)
    b = np.random.rand(5,)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    primals = (a_tensor, b_tensor)
    tangents = (deepnet.ones_like(a_tensor), deepnet.ones_like(b_tensor))

    def func(a, b):
        return f.sum(f.permute(f.sine(f.add(a, b)), (1, 0)), dims=1)

    result_tensor, output_tangent = jvp(primals, tangents, func, use_graph=True)
    result_tensor.backward(output_tangent)
    expected = np.sum(np.transpose(np.sin(np.add(a, b))), axis=1)
    np.testing.assert_allclose(
        result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim() == output_tangent.dim()
    for primal in primals:
        assert primal.grad is not None

def test_grad_add_sub():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)
    c = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    c_tensor = deepnet.tensor(c, use_grad=True)
    result_tensor = f.sub(f.add(a_tensor, b_tensor), c_tensor)

    output_grad = deepnet.ones_like(result_tensor)
    partial_derivatives = grad((a_tensor, b_tensor, c_tensor), result_tensor, output_grad)
    result_tensor.backward(output_grad)

    for primal, partial_derivative in zip((a_tensor, b_tensor, c_tensor), partial_derivatives):
        assert np.allclose(primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5)

def test_grad_mul_div():
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)
    c = np.random.rand(2, 2)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    c_tensor = deepnet.tensor(c, use_grad=True)
    result_tensor = f.div(f.mul(a_tensor, b_tensor), c_tensor)

    output_grad = deepnet.ones_like(result_tensor) 
    partial_derivatives = grad((a_tensor, b_tensor, c_tensor), result_tensor, output_grad)
    result_tensor.backward(output_grad)

    for primal, partial_derivative in zip((a_tensor, b_tensor, c_tensor), partial_derivatives):
        assert np.allclose(primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5)

def test_grad_matmul_sum():
    a = np.random.rand(2, 3)
    b = np.random.rand(3, 2)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.sum(f.matmul(a_tensor, b_tensor), dims=0)

    output_grad = deepnet.ones_like(result_tensor) 
    partial_derivatives = grad((a_tensor, b_tensor), result_tensor, output_grad)
    result_tensor.backward(output_grad)

    for primal, partial_derivative in zip((a_tensor, b_tensor), partial_derivatives):
        assert np.allclose(primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5)

def test_grad_add_mul_broadcast():
    a = np.random.rand(2, 4)
    b = np.random.rand(4,)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.mul(f.add(a_tensor, b_tensor), b_tensor)

    output_grad = deepnet.ones_like(result_tensor) 
    partial_derivatives = grad((a_tensor, b_tensor), result_tensor, output_grad)
    result_tensor.backward(output_grad)

    for primal, partial_derivative in zip((a_tensor, b_tensor), partial_derivatives):
        assert np.allclose(primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5)

def test_grad_nested_operations_broadcast():
    a = np.random.rand(3, 3)
    b = np.random.rand(3,)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)

    result_tensor = f.cosine(f.div(f.add(a_tensor, b_tensor), f.sine(b_tensor)))
    output_grad = deepnet.ones_like(result_tensor) 
    partial_derivatives = grad((a_tensor, b_tensor), result_tensor, output_grad)
    result_tensor.backward(output_grad)

    for primal, partial_derivative in zip((a_tensor, b_tensor), partial_derivatives):
        assert np.allclose(primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5)


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

    # Basic JVP Tests

    test_jvp_no_graph()
    test_jvp_with_graph()

    # Composition Function JVP Tests

    test_jvp_add_sub()
    test_jvp_mul_div()
    test_jvp_matmul_sum()

    # Broadcasting Function JVP Tests

    test_jvp_add_mul_broadcast()
    test_jvp_nested_operations_broadcast()
    test_jvp_sine_permute_sum_broadcast()

    # Grad Tests

    test_grad_add_sub()
    test_grad_mul_div()
    test_grad_matmul_sum()
    test_grad_add_mul_broadcast()
    test_grad_nested_operations_broadcast()

    print("All tests passed")


if __name__ == "__main__":
    main()
