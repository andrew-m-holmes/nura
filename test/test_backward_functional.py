import nura
import nura.functional as f
import numpy as np


def test_add_scalar():
    a, b = 1.0, 3.0
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.add(a_tensor, b_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x, y):
        return x + y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)
    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_add_vector_backward():
    a = np.random.rand(3)
    b = np.random.rand(3)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.add(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x + y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_add_matrix_backward():
    a = np.random.rand(4, 3)
    b = np.random.rand(4, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.add(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x + y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_add_tensor_backward():
    a = np.random.rand(2, 5, 3)
    b = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.add(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x + y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_add_broadcast_backward():
    a = np.random.rand(5, 3, 2)
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.add(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x + y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = np.sum(
        (func(a, b + h) - func(a, b - h)) / (2 * h), axis=(0, 2)
    ).reshape(b.shape)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data,
        expected_grad_b,
        rtol=1e-7,
        atol=1e-7,
    )


def test_add_different_types_backward():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.add(a_tensor, 3)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x + 3

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_add_different_types_reversed_backward():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = 3 + a_tensor
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 3 + x

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sub_scalar_backward():
    a, b = np.array(2.0), np.array(1.0)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.sub(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x - y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_sub_vector_backward():
    a = np.random.rand(5)
    b = np.random.rand(5)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.sub(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x - y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_sub_matrix_backward():
    a = np.random.rand(3, 4)
    b = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.sub(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x - y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_sub_tensor_backward():
    a = np.random.rand(2, 4, 3)
    b = np.random.rand(2, 4, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.sub(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x - y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_sub_broadcast_backward():
    a = np.random.rand(4, 3, 2)
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.sub(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x - y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = np.sum(
        (func(a, b + h) - func(a, b - h)) / (2 * h), axis=(0, 2)
    ).reshape(b.shape)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_sub_different_types_backward():
    a = np.random.rand(3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.sub(a_tensor, 2)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x - 2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sub_different_types_reversed_backward():
    a = np.random.rand(3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = 2 - a_tensor
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 2 - x

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_mul_scalar_backward():
    a, b = np.array(2.0), np.array(3.0)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.mul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x * y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_mul_vector_backward():
    a = np.random.rand(4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.mul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x * y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_mul_matrix_backward():
    a = np.random.rand(2, 5)
    b = np.random.rand(2, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.mul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x * y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_mul_tensor_backward():
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.mul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x * y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_mul_broadcast_backward():
    a = np.random.rand(3, 4, 2)
    b = np.random.rand(4, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.mul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x * y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = np.sum(
        (func(a, b + h) - func(a, b - h)) / (2 * h), axis=(0, 2)
    ).reshape(b.shape)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_mul_different_types_backward():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.mul(a_tensor, 2)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x * 2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_mul_different_types_reversed_backward():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = 2 * a_tensor
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 2 * x

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_div_scalar_backward():
    a, b = np.array(6.0), np.array(2.0)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.div(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x / y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_div_vector_backward():
    a = np.random.rand(5) + 1e-7
    b = np.random.rand(5) + 1e-7
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.div(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x / y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_div_matrix_backward():
    a = np.random.rand(3, 4) + 1e-7
    b = np.random.rand(3, 4) + 1e-7
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.div(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x / y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_div_tensor_backward():
    a = np.random.rand(2, 4, 3) + 1e-7
    b = np.random.rand(2, 4, 3) + 1e-7
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.div(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x / y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_div_broadcast_backward():
    a = np.random.rand(4, 3, 2) + 1e-7
    b = np.random.rand(3, 1) + 1e-7
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.div(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x / y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)
    expected_grad_b = np.sum(
        (func(a, b + h) - func(a, b - h)) / (2 * h), axis=(0, 2)
    ).reshape(b.shape)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_div_different_types_backward():
    a = np.random.rand(4) + 1e-7
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.div(a_tensor, 2)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x / 2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_div_different_types_reversed_backward():
    a = np.random.rand(4) + 1e-7
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = 2 / a_tensor
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 2 / x

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_dot_backward():
    a = np.random.rand(4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.dot(a_tensor, b_tensor)
    result_tensor.backward()

    expected_grad_a = b
    expected_grad_b = a

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_dot_method_backward():
    a = np.random.rand(4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = a_tensor.dot(b_tensor)
    result_tensor.backward()

    expected_grad_a = b
    expected_grad_b = a

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_matmul_matrix_matrix_backward():
    a = np.random.rand(3, 4)
    b = np.random.rand(4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.dot(np.ones((3, 5)), b.T)
    expected_grad_b = np.dot(a.T, np.ones((3, 5)))

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_matmul_tensor_tensor_backward():
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(2, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.matmul(np.ones((2, 3, 5)), b.swapaxes(-1, -2))
    expected_grad_b = np.matmul(a.swapaxes(-1, -2), np.ones((2, 3, 5)))

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_matmul_higher_rank_tensor_tensor_backward():
    a = np.random.rand(2, 3, 4, 6)
    b = np.random.rand(2, 3, 6, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.matmul(np.ones((2, 3, 4, 5)), b.swapaxes(-1, -2))
    expected_grad_b = np.matmul(a.swapaxes(-1, -2), np.ones((2, 3, 4, 5)))

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_matmul_tensor_vector_backward():
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.einsum("ij,k->ijk", np.ones((2, 3)), b)
    expected_grad_b = np.einsum("ijk,ij->k", a, np.ones((2, 3)))

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_matmul_vector_tensor_backward():
    a = np.random.rand(4)
    b = np.random.rand(2, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.einsum("ijk,ik->j", b, np.ones((2, 5)))
    expected_grad_b = np.einsum("ik,j->ijk", np.ones((2, 5)), a)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_matmul_higher_rank_tensor_vector_backward():
    a = np.random.rand(2, 3, 6, 4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.einsum("ijk,l->ijkl", np.ones((2, 3, 6)), b)
    expected_grad_b = np.einsum("ijkl,ijk->l", a, np.ones((2, 3, 6)))

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_matmul_vector_higher_rank_tensor_backward():
    a = np.random.rand(4)
    b = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.einsum("ijkl,ijl->k", b, np.ones((2, 3, 5)))
    expected_grad_b = np.einsum("ijl,k->ijkl", np.ones((2, 3, 5)), a)

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_matmul_operator_backward():
    a = np.random.rand(3, 4)
    b = np.random.rand(4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = a_tensor @ b_tensor
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.dot(np.ones((3, 5)), b.T)
    expected_grad_b = np.dot(a.T, np.ones((3, 5)))

    assert a_tensor.grad is not None
    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_pow_scalar_backward():
    a, b = 2.0, 3.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pow(a_tensor, b)
    result_tensor.backward()
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_pow_vector_backward():
    a = np.random.rand(3)
    b = 3.0
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=False)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_pow_matrix_backward():
    a = np.random.rand(4, 3)
    b = 2.0
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=False)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_pow_tensor_backward():
    a = np.random.rand(2, 5, 3)
    b = 2.0
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=False)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_pow_broadcast_backward():
    a = np.random.rand(5, 3, 2)
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=False)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_pow_different_types_backward():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pow(a_tensor, 3)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x ** 3

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)


def test_pow_scalar_b_backward():
    a, b = 2.0, 3.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7)

def test_pow_vector_b_backward():
    a = np.abs(np.random.rand(3))
    b = np.random.rand(3)
    a_tensor = nura.tensor(a, usegrad=False)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7)

def test_pow_matrix_b_backward():
    a = np.abs(np.random.rand(4, 3))
    b = np.random.rand(4, 3)
    a_tensor = nura.tensor(a, usegrad=False)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7)

def test_pow_tensor_b_backward():
    a = np.abs(np.random.rand(2, 5, 3))
    b = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=False)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7)

def test_pow_broadcast_b_backward():
    a = np.abs(np.random.rand(5, 3, 2))
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a, usegrad=False)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_b = np.sum((func(a, b + h) - func(a, b - h)) / (2 * h), axis=(0, 2)).reshape(b.shape)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7)

def test_pow_different_types_b_backward():
    a = 3
    b = np.random.rand(4)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = a ** b_tensor
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7)

def test_pow_operator_b_backward():
    a = np.abs(np.random.rand(4))
    b = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=False)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = a_tensor ** b_tensor
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x ** y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7)

def test_square_scalar_backward():
    a = 3.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return x ** 2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_square_vector_backward():
    a = np.random.rand(3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x ** 2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_square_matrix_backward():
    a = np.random.rand(4, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x ** 2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_square_tensor_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x ** 2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_square_method_tensor_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.square()
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x ** 2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)


def test_sqrt_scalar_backward():
    a = 3.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sqrt(a_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.sqrt(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sqrt_vector_backward():
    a = np.random.rand(3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sqrt(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.sqrt(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sqrt_matrix_backward():
    a = np.random.rand(4, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sqrt(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.sqrt(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sqrt_tensor_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sqrt(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.sqrt(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sqrt_method_tensor_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.sqrt()
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.sqrt(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)


def test_exp_scalar_backward():
    a = 2.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.exp(a_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.exp(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_exp_vector_backward():
    a = np.random.rand(3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.exp(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.exp(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_exp_matrix_backward():
    a = np.random.rand(4, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.exp(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.exp(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_exp_tensor_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.exp(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.exp(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_exp_method_tensor_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.exp()
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.exp(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)


def test_log_scalar_backward():
    a = 4.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.log(a_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.log(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_log_vector_backward():
    a = np.abs(np.random.rand(5))
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.log(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.log(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_log_matrix_backward():
    a = np.abs(np.random.rand(4, 3))
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.log(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.log(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_log_tensor_backward():
    a = np.abs(np.random.rand(2, 4, 3))
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.log(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.log(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_log_method_tensor_backward():
    a = np.abs(np.random.rand(3, 2, 4))
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.log()
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.log(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)


def test_sin_scalar_backward():
    a = 3.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sin(a_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.sin(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sin_vector_backward():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sin(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.sin(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sin_matrix_backward():
    a = np.random.rand(4, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sin(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.sin(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sin_tensor_backward():
    a = np.random.rand(2, 4, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sin(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.sin(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sin_method_tensor_backward():
    a = np.random.rand(3, 2, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.sin()
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.sin(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)


def test_cos_scalar_backward():
    a = 1.5
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.cos(a_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.cos(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_cos_vector_backward():
    a = np.random.rand(6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.cos(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.cos(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_cos_matrix_backward():
    a = np.random.rand(4, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.cos(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.cos(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_cos_tensor_backward():
    a = np.random.rand(3, 3, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.cos(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.cos(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_cos_method_tensor_backward():
    a = np.random.rand(5, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.cos()
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.cos(x)

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_scalar_backward():
    a = 7.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_vector_backward():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_method_backward():
    a = np.random.rand(8, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.sum()
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_tuple_backward():
    a = np.random.rand(2, 4, 7)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=(0, 1))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_tuple_keepdims_true_backward():
    a = np.random.rand(1, 3, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=(0, 2), keepdims=True)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_tuple_keepdims_false_backward():
    a = np.random.rand(4, 2, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=(1, 2), keepdims=False)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_0_shape_1_backward():
    a = np.random.rand(5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_1_shape_1_backward():
    a = np.random.rand(4, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_0_shape_2_backward():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_1_shape_2_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_2_shape_2_backward():
    a = np.random.rand(4, 2, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_0_shape_3_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_1_shape_3_backward():
    a = np.random.rand(3, 4, 2, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_2_shape_3_backward():
    a = np.random.rand(4, 3, 5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_sum_dim_3_shape_3_backward():
    a = np.random.rand(2, 4, 3, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=3)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_scalar_backward():
    a = 5.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_vector_backward():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.max()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.max()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.max()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_method_backward():
    a = np.random.rand(5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.max()
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.max()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_tuple_backward():
    a = np.random.rand(2, 4, 7)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=(0, 1))
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=(0, 1), keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_tuple_keepdims_true_backward():
    a = np.random.rand(1, 3, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=(0, 2), keepdims=True)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=(0, 2), keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_0_shape_1_backward():
    a = np.random.rand(5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_1_shape_1_backward():
    a = np.random.rand(4, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_0_shape_2_backward():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_1_shape_2_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_2_shape_2_backward():
    a = np.random.rand(4, 2, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=2, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_0_shape_3_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_1_shape_3_backward():
    a = np.random.rand(3, 4, 2, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_2_shape_3_backward():
    a = np.random.rand(4, 3, 5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=2, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_max_dim_3_shape_3_backward():
    a = np.random.rand(2, 4, 3, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=3)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=3, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_scalar_backward():
    a = 5.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_vector_backward():
    a = np.array([3.0, 2.0, 1.0, 4.0])
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.min()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.min()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.min()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_method_backward():
    a = np.random.rand(5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.min()
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.min()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_tuple_backward():
    a = np.random.rand(2, 4, 7)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=(0, 1))
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=(0, 1), keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_tuple_keepdims_true_backward():
    a = np.random.rand(1, 3, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=(0, 2), keepdims=True)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=(0, 2), keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_0_shape_1_backward():
    a = np.random.rand(5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_1_shape_1_backward():
    a = np.random.rand(4, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_0_shape_2_backward():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_1_shape_2_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_2_shape_2_backward():
    a = np.random.rand(4, 2, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=2, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_0_shape_3_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_1_shape_3_backward():
    a = np.random.rand(3, 4, 2, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_2_shape_3_backward():
    a = np.random.rand(4, 3, 5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=2, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)

def test_min_dim_3_shape_3_backward():
    a = np.random.rand(2, 4, 3, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=3)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=3, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7)
