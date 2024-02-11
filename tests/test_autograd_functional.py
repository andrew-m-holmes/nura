import deepnet
import deepnet.functional as f
from deepnet.autograd.functional import vjp, jvp, grad
import numpy as np


def test_vjp_basic():
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

    result_tensor, cotangents = vjp(primals, cotangent, func)
    expected = np.add(np.matmul(a, b), c)
    assert np.allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    for primal, cotangent in zip(primals, cotangents):
        assert primal.dim == cotangent.dim


def test_vjp_with_graph():
    a = np.random.rand(5, 3)
    b = np.random.rand(3, 7)
    c = np.random.rand(1)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    c_tensor = deepnet.tensor(c, usegrad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    cotangent = deepnet.ones((5, 7))

    def func(a, b, c):
        return f.add(f.matmul(a, b), c)

    result_tensor, cotangents = vjp(primals, cotangent, func)
    expected = np.add(np.matmul(a, b), c)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)


def test_vjp_add_sub():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    c = np.random.rand(4, 4)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    c_tensor = deepnet.tensor(c, usegrad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    cotangent = deepnet.ones((4, 4))

    def func(a, b, c):
        return f.sub(f.add(a, b), c)

    result_tensor, cotangents = vjp(primals, cotangent, func)
    expected = np.subtract(np.add(a, b), c)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)


def test_vjp_mul_div():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)
    c = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    c_tensor = deepnet.tensor(c, usegrad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    cotangent = deepnet.ones((5, 5))

    def func(a, b, c):
        return f.div(f.mul(a, b), c)

    result_tensor, cotangents = vjp(primals, cotangent, func)
    expected = np.divide(np.multiply(a, b), c)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)


def test_vjp_matmul_sum():
    a = np.random.rand(3, 4)
    b = np.random.rand(4, 3)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    primals = (a_tensor, b_tensor)
    cotangent = deepnet.ones((3,))

    def func(a, b):
        return f.sum(f.matmul(a, b), dim=0)

    result_tensor, cotangents = vjp(primals, cotangent, func)
    expected = np.sum(np.matmul(a, b), axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)


def test_vjp_add_mul_broadcast():
    a = np.random.rand(3, 4)
    b = np.random.rand(
        4,
    )

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    primals = (a_tensor, b_tensor)
    cotangent = deepnet.ones((3, 4))

    def func(a, b):
        return f.mul(f.add(a, b), b)

    result_tensor, cotangents = vjp(primals, cotangent, func)
    expected = np.multiply(np.add(a, b), b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)


def test_vjp_nested_operations_broadcast():
    a = np.random.rand(5, 6)
    b = np.random.rand(
        6,
    )

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    primals = (a_tensor, b_tensor)
    cotangent = deepnet.ones((5, 6))

    def func(a, b):
        return f.cos(f.div(f.add(a, b), f.sin(b)))

    result_tensor, cotangents = vjp(primals, cotangent, func)
    expected = np.cos(np.divide(np.add(a, b), np.sin(b)))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)


def test_vjp_matmul_add_broadcast():
    a = np.random.rand(4, 3)
    b = np.random.rand(3, 5)
    c = np.random.rand(
        5,
    )

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    c_tensor = deepnet.tensor(c, usegrad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    cotangent = deepnet.ones((4, 5))

    def func(a, b, c):
        return f.add(f.matmul(a, b), c)

    result_tensor, cotangents = vjp(primals, cotangent, func)
    expected = np.add(np.matmul(a, b), c)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)


def test_jvp_basic():
    a = np.random.rand(6, 4)
    b = np.random.rand(10, 4)
    c = np.random.rand(1)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    c_tensor = deepnet.tensor(c)
    primals = (a_tensor, b_tensor, c_tensor)
    tangents = (
        deepnet.oneslike(a_tensor),
        deepnet.oneslike(b_tensor),
        deepnet.oneslike(c_tensor),
    )

    def func(a, b, c):
        return f.add(f.matmul(a, b.transpose()), c)

    result_tensor, output_tangent = jvp(primals, tangents, func)
    expected = np.add(np.matmul(a, b.T), c)
    assert np.allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim == output_tangent.dim


def test_jvp_with_graph():
    a = np.random.rand(5, 3)
    b = np.random.rand(3, 7)
    c = np.random.rand(1)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    c_tensor = deepnet.tensor(c, usegrad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    tangents = (
        deepnet.oneslike(a_tensor),
        deepnet.oneslike(b_tensor),
        deepnet.oneslike(c_tensor),
    )

    def func(a, b, c):
        return f.add(f.matmul(a, b), c)

    result_tensor, output_tangent = jvp(primals, tangents, func)
    expected = np.add(np.matmul(a, b), c)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim == output_tangent.dim


def test_jvp_add_sub():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    c = np.random.rand(4, 4)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    c_tensor = deepnet.tensor(c, usegrad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    tangents = (
        deepnet.oneslike(a_tensor),
        deepnet.oneslike(b_tensor),
        deepnet.oneslike(c_tensor),
    )

    def func(a, b, c):
        return f.sub(f.add(a, b), c)

    result_tensor, output_tangent = jvp(primals, tangents, func)
    expected = np.subtract(np.add(a, b), c)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim == output_tangent.dim


def test_jvp_mul_div():
    a = np.random.rand(7, 7)
    b = np.random.rand(7, 7)
    c = np.random.rand(7, 7)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    c_tensor = deepnet.tensor(c, usegrad=True)
    primals = (a_tensor, b_tensor, c_tensor)
    tangents = (
        deepnet.oneslike(a_tensor),
        deepnet.oneslike(b_tensor),
        deepnet.oneslike(c_tensor),
    )

    def func(a, b, c):
        return f.div(f.mul(a, b), c)

    result_tensor, output_tangent = jvp(primals, tangents, func)
    expected = np.divide(np.multiply(a, b), c)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim == output_tangent.dim


def test_jvp_matmul_sum():
    a = np.random.rand(3, 8)
    b = np.random.rand(8, 3)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    primals = (a_tensor, b_tensor)
    tangents = (deepnet.oneslike(a_tensor), deepnet.oneslike(b_tensor))

    def func(a, b):
        return f.sum(f.matmul(a, b), dim=0)

    result_tensor, output_tangent = jvp(primals, tangents, func)
    expected = np.sum(np.matmul(a, b), axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim == output_tangent.dim


def test_jvp_add_mul_broadcast():
    a = np.random.rand(3, 4)
    b = np.random.rand(
        4,
    )

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    primals = (a_tensor, b_tensor)
    tangents = (deepnet.oneslike(a_tensor), deepnet.oneslike(b_tensor))

    def func(a, b):
        return f.mul(f.add(a, b), b)

    result_tensor, output_tangent = jvp(primals, tangents, func)
    expected = np.multiply(np.add(a, b), b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim == output_tangent.dim


def test_jvp_nested_operations_broadcast():
    a = np.random.rand(5, 4)
    b = np.random.rand(
        4,
    )

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    primals = (a_tensor, b_tensor)
    tangents = (deepnet.oneslike(a_tensor), deepnet.oneslike(b_tensor))

    def func(a, b):
        return f.cos(f.div(f.add(a, b), f.sin(b)))

    result_tensor, output_tangent = jvp(primals, tangents, func)
    expected = np.cos(np.divide(np.add(a, b), np.sin(b)))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim == output_tangent.dim


def test_jvp_sin_permute_sum_broadcast():
    a = np.random.rand(4, 5)
    b = np.random.rand(
        5,
    )

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    primals = (a_tensor, b_tensor)
    tangents = (deepnet.oneslike(a_tensor), deepnet.oneslike(b_tensor))

    def func(a, b):
        return f.sum(f.permute(f.sin(f.add(a, b)), (1, 0)), dim=1)

    result_tensor, output_tangent = jvp(primals, tangents, func)
    expected = np.sum(np.transpose(np.sin(np.add(a, b))), axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-5, atol=1e-5)
    assert result_tensor.dim == output_tangent.dim


def test_grad_add_sub():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)
    c = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    c_tensor = deepnet.tensor(c, usegrad=True)
    result_tensor = f.sub(f.add(a_tensor, b_tensor), c_tensor)

    output_grad = deepnet.oneslike(result_tensor)
    partial_derivatives = grad(
        (a_tensor, b_tensor, c_tensor), result_tensor, output_grad
    )
    result_tensor.backward(output_grad)

    for primal, partial_derivative in zip(
        (a_tensor, b_tensor, c_tensor), partial_derivatives
    ):
        assert np.allclose(
            primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5
        )


def test_grad_mul_div():
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)
    c = np.random.rand(2, 2)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    c_tensor = deepnet.tensor(c, usegrad=True)
    result_tensor = f.div(f.mul(a_tensor, b_tensor), c_tensor)

    output_grad = deepnet.oneslike(result_tensor)
    partial_derivatives = grad(
        (a_tensor, b_tensor, c_tensor), result_tensor, output_grad
    )
    result_tensor.backward(output_grad)

    for primal, partial_derivative in zip(
        (a_tensor, b_tensor, c_tensor), partial_derivatives
    ):
        assert np.allclose(
            primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5
        )


def test_grad_matmul_sum():
    a = np.random.rand(2, 3)
    b = np.random.rand(3, 2)

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    result_tensor = f.sum(f.matmul(a_tensor, b_tensor), dim=0)

    output_grad = deepnet.oneslike(result_tensor)
    partial_derivatives = grad((a_tensor, b_tensor), result_tensor, output_grad)
    result_tensor.backward(output_grad)

    for primal, partial_derivative in zip((a_tensor, b_tensor), partial_derivatives):
        assert np.allclose(
            primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5
        )


def test_grad_add_mul_broadcast():
    a = np.random.rand(2, 4)
    b = np.random.rand(
        4,
    )

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    result_tensor = f.mul(f.add(a_tensor, b_tensor), b_tensor)

    output_grad = deepnet.oneslike(result_tensor)
    partial_derivatives = grad((a_tensor, b_tensor), result_tensor, output_grad)
    result_tensor.backward(output_grad)

    for primal, partial_derivative in zip((a_tensor, b_tensor), partial_derivatives):
        assert np.allclose(
            primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5
        )


def test_grad_nested_operations_broadcast():
    a = np.random.rand(3, 3)
    b = np.random.rand(
        3,
    )

    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)

    result_tensor = f.cos(f.div(f.add(a_tensor, b_tensor), f.sin(b_tensor)))
    output_grad = deepnet.oneslike(result_tensor)
    partial_derivatives = grad((a_tensor, b_tensor), result_tensor, output_grad)
    result_tensor.backward(output_grad)

    for primal, partial_derivative in zip((a_tensor, b_tensor), partial_derivatives):
        assert np.allclose(
            primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5
        )


def test_grad_cos_scalar():
    a = np.random.rand()
    a_tensor = deepnet.tensor(a, usegrad=True)
    result_tensor = f.cos(a_tensor)

    partial_derivatives, *_= grad(a_tensor, result_tensor)
    result_tensor.backward()
    print(partial_derivatives)

    assert np.allclose(
        a_tensor.grad.data, partial_derivatives.data, rtol=1e-5, atol=1e-5
    )


def test_grad_sin_vector():
    a = np.random.rand(5)
    a_tensor = deepnet.tensor(a, usegrad=True)
    result_tensor = f.sin(a_tensor)

    partial_derivatives, *_= grad(a_tensor, result_tensor, deepnet.oneslike(result_tensor))
    result_tensor.backward(deepnet.oneslike(result_tensor))
    print(partial_derivatives)

    assert np.allclose(
        a_tensor.grad.data, partial_derivatives.data, rtol=1e-5, atol=1e-5
    )


def test_grad_div_large_tensor():
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    a_tensor = deepnet.tensor(a, usegrad=True)
    b_tensor = deepnet.tensor(b, usegrad=True)
    result_tensor = f.div(a_tensor, b_tensor)

    partial_derivatives = grad(
        (a_tensor, b_tensor), result_tensor, deepnet.oneslike(result_tensor)
    )
    result_tensor.backward(deepnet.oneslike(result_tensor))

    for primal, partial_derivative in zip((a_tensor, b_tensor), partial_derivatives):
        assert np.allclose(
            primal.grad.data, partial_derivative.data, rtol=1e-5, atol=1e-5
        )


def test_grad_permute_complex_tensor():
    a = np.random.rand(3, 4, 5)
    a_tensor = deepnet.tensor(a, usegrad=True)
    result_tensor = deepnet.permute(a_tensor, (2, 0, 1))

    partial_derivatives, *_ = grad(a_tensor, result_tensor, deepnet.oneslike(result_tensor))
    result_tensor.backward(deepnet.oneslike(result_tensor))
    print(partial_derivatives)

    assert np.allclose(
        a_tensor.grad.data, partial_derivatives.data, rtol=1e-5, atol=1e-5
    )


def main():

    # Basic VJP Tests

    test_vjp_basic()
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

    test_jvp_basic()
    test_jvp_with_graph()

    # Composition Function JVP Tests

    test_jvp_add_sub()
    test_jvp_mul_div()
    test_jvp_matmul_sum()

    # Broadcasting Function JVP Tests

    test_jvp_add_mul_broadcast()
    test_jvp_nested_operations_broadcast()
    test_jvp_sin_permute_sum_broadcast()

    # Grad Tests

    test_grad_add_sub()
    test_grad_mul_div()
    test_grad_matmul_sum()
    test_grad_add_mul_broadcast()
    test_grad_nested_operations_broadcast()

    print("All tests passed")


if __name__ == "__main__":
    main()
