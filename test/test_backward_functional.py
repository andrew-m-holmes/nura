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
        return x**y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pow_vector_backward():
    a = np.random.rand(3)
    b = 3.0
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=False)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pow_matrix_backward():
    a = np.random.rand(4, 3)
    b = 2.0
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=False)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pow_tensor_backward():
    a = np.random.rand(2, 5, 3)
    b = 2.0
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=False)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pow_broadcast_backward():
    a = np.random.rand(5, 3, 2)
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=False)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_a = (func(a + h, b) - func(a - h, b)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pow_different_types_backward():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pow(a_tensor, 3)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x**3

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pow_scalar_b_backward():
    a, b = 2.0, 3.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_pow_vector_b_backward():
    a = np.abs(np.random.rand(3))
    b = np.random.rand(3)
    a_tensor = nura.tensor(a, usegrad=False)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_pow_matrix_b_backward():
    a = np.abs(np.random.rand(4, 3))
    b = np.random.rand(4, 3)
    a_tensor = nura.tensor(a, usegrad=False)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_pow_tensor_b_backward():
    a = np.abs(np.random.rand(2, 5, 3))
    b = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=False)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_pow_broadcast_b_backward():
    a = np.abs(np.random.rand(5, 3, 2))
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a, usegrad=False)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_b = np.sum(
        (func(a, b + h) - func(a, b - h)) / (2 * h), axis=(0, 2)
    ).reshape(b.shape)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_pow_different_types_b_backward():
    a = 3
    b = np.random.rand(4)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = a**b_tensor
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_pow_operator_b_backward():
    a = np.abs(np.random.rand(4))
    b = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=False)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = a_tensor**b_tensor
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x, y):
        return x**y

    expected_grad_b = (func(a, b + h) - func(a, b - h)) / (2 * h)

    assert b_tensor.grad is not None
    np.testing.assert_allclose(
        b_tensor.grad.data, expected_grad_b, rtol=1e-7, atol=1e-7
    )


def test_square_scalar_backward():
    a = 3.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return x**2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_square_vector_backward():
    a = np.random.rand(3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x**2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_square_matrix_backward():
    a = np.random.rand(4, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x**2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_square_tensor_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x**2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_square_method_tensor_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.square()
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return x**2

    expected_grad_a = (func(a + h) - func(a - h)) / (2 * h)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


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
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_scalar_backward():
    a = 7.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_vector_backward():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_method_backward():
    a = np.random.rand(8, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.sum()
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_tuple_backward():
    a = np.random.rand(2, 4, 7)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=(0, 1))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_tuple_keepdims_true_backward():
    a = np.random.rand(1, 3, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=(0, 2), keepdims=True)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_tuple_keepdims_false_backward():
    a = np.random.rand(4, 2, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=(1, 2), keepdims=False)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_0_shape_1_backward():
    a = np.random.rand(5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_1_shape_1_backward():
    a = np.random.rand(4, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_0_shape_2_backward():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_1_shape_2_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_2_shape_2_backward():
    a = np.random.rand(4, 2, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_0_shape_3_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_1_shape_3_backward():
    a = np.random.rand(3, 4, 2, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_2_shape_3_backward():
    a = np.random.rand(4, 3, 5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_sum_dim_3_shape_3_backward():
    a = np.random.rand(2, 4, 3, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sum(a_tensor, dim=3)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_scalar_backward():
    a = 5.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_vector_backward():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.max()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.max()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.max()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_method_backward():
    a = np.random.rand(5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.max()
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.max()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_tuple_backward():
    a = np.random.rand(2, 4, 7)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=(0, 1))
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=(0, 1), keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_tuple_keepdims_true_backward():
    a = np.random.rand(1, 3, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=(0, 2), keepdims=True)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=(0, 2), keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_0_shape_1_backward():
    a = np.random.rand(5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_1_shape_1_backward():
    a = np.random.rand(4, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_0_shape_2_backward():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_1_shape_2_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_2_shape_2_backward():
    a = np.random.rand(4, 2, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=2, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_0_shape_3_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_1_shape_3_backward():
    a = np.random.rand(3, 4, 2, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_2_shape_3_backward():
    a = np.random.rand(4, 3, 5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=2, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_max_dim_3_shape_3_backward():
    a = np.random.rand(2, 4, 3, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.max(a_tensor, dim=3)
    result_tensor.backward(nura.oneslike(result_tensor))

    max_vals = a.max(axis=3, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == max_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_scalar_backward():
    a = 5.0
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor)
    result_tensor.backward()

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_vector_backward():
    a = np.array([3.0, 2.0, 1.0, 4.0])
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.min()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.min()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.min()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_method_backward():
    a = np.random.rand(5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.min()
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == a.min()] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_tuple_backward():
    a = np.random.rand(2, 4, 7)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=(0, 1))
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=(0, 1), keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_tuple_keepdims_true_backward():
    a = np.random.rand(1, 3, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=(0, 2), keepdims=True)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=(0, 2), keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_0_shape_1_backward():
    a = np.random.rand(5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_1_shape_1_backward():
    a = np.random.rand(4, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_0_shape_2_backward():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_1_shape_2_backward():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_2_shape_2_backward():
    a = np.random.rand(4, 2, 6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=2, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_0_shape_3_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=0)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=0, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_1_shape_3_backward():
    a = np.random.rand(3, 4, 2, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=1)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=1, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_2_shape_3_backward():
    a = np.random.rand(4, 3, 5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=2)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=2, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_min_dim_3_shape_3_backward():
    a = np.random.rand(2, 4, 3, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.min(a_tensor, dim=3)
    result_tensor.backward(nura.oneslike(result_tensor))

    min_vals = a.min(axis=3, keepdims=True)
    expected_grad_a = np.zeros_like(a)
    expected_grad_a[a == min_vals] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


# TODO tests for mean var and std


def test_transpose_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.transpose(a_tensor, 0, 1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_transpose_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.transpose(a_tensor, 1, 2)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_transpose_higher_rank_tensor_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.transpose(a_tensor, -1, -3)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_transpose_method_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.transpose(0, 1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_transpose_method_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.transpose(1, -1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_transpose_method_higher_rank_tensor_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.transpose(-4, 2)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_permute_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.permute(a_tensor, (1, 0))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_permute_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.permute(a_tensor, (2, 0, 1))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_permute_higher_rank_tensor_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.permute(a_tensor, (3, 2, 1, 0))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_permute_method_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.permute((1, 0))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_permute_method_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.permute((2, -3, 1))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_permute_method_higher_rank_tensor_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.permute((3, -2, 1, -4))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_squeeze_scalar_backward():
    a = np.array(3.0)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.squeeze(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_squeeze_vector_backward():
    a = np.random.rand(1)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.squeeze(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_squeeze_matrix_backward():
    a = np.random.rand(3, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.squeeze(a_tensor, 1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_squeeze_tensor_backward():
    a = np.random.rand(2, 1, 3, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.squeeze(a_tensor, (1, 3))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_squeeze_method_higher_order_tensor_backward():
    a = np.random.rand(1, 5, 1, 2, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.squeeze((0, -1))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_squeeze_higher_order_tensor_no_dim_backward():
    a = np.random.rand(1, 5, 1, 2, 1)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.squeeze(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_unsqueeze_scalar_backward():
    a = np.array(3.0)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.unsqueeze(a_tensor, 0)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_unsqueeze_vector_backward():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.unsqueeze(a_tensor, 1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_unsqueeze_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.unsqueeze(a_tensor, -1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_unsqueeze_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.unsqueeze(a_tensor, -2)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_unsqueeze_higher_order_tensor_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.unsqueeze(a_tensor, 1)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_unsqueeze_method_higher_order_tensor_backward():
    a = np.random.rand(3, 1, 4, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.unsqueeze(-4)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_unsqueeze_mixed_indices_backward():
    a = np.random.rand(3, 4, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.unsqueeze(a_tensor, (-3, 3))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_reshape_scalar_backward():
    a = np.array(7.0)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.reshape(a_tensor, (1, 1, 1))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_reshape_vector_backward():
    a = np.random.rand(6)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.reshape(a_tensor, (3, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_reshape_matrix_backward():
    a = np.random.rand(4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.reshape(a_tensor, (20,))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_reshape_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.reshape(a_tensor, (4, 6))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_reshape_higher_order_tensor_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.reshape(a_tensor, (6, 20))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_reshape_method_higher_order_tensor_backward():
    a = np.random.rand(3, 4, 5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.reshape((5, 24))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_reshape_with_negative_dimension_backward_1():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.reshape(a_tensor, (-1, 6))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_reshape_with_negative_dimension_backward_2():
    a = np.random.rand(6, 2, 3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.reshape(a_tensor, (3, -1))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_abs_scalar_backward():
    a = np.array(-7.0)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.abs(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.array(-1.0)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_abs_vector_backward():
    a = np.array([-1.0, 2.0, -3.0, 4.0])
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.abs(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.where(a < 0, -1.0, 1.0)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_abs_matrix_backward():
    a = np.random.randn(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.abs(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.where(a < 0, -1.0, 1.0)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_abs_tensor_backward():
    a = np.random.randn(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.abs(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.where(a < 0, -1.0, 1.0)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_abs_higher_order_tensor_backward():
    a = np.random.randn(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.abs(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.where(a < 0, -1.0, 1.0)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_abs_method_higher_order_tensor_backward():
    a = np.random.randn(3, 4, 5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.abs()
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.where(a < 0, -1.0, 1.0)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pos_scalar_backward():
    a = np.array(3.0)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pos(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pos_vector_backward():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pos(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pos_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pos(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pos_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pos(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pos_higher_order_tensor_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pos(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_pos_operator_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = +a_tensor
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_neg_scalar_backward():
    a = np.array(4.0)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.neg(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = -np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_neg_vector_backward():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.neg(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = -np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_neg_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.neg(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = -np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_neg_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.neg(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = -np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_neg_higher_order_tensor_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.neg(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = -np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_neg_operator_backward():
    a = np.random.rand(3, 4, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = -a_tensor
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = -np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_clone_scalar_backward():
    a = np.array(4.0)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.clone(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_clone_vector_backward():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.clone(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_clone_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.clone(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_clone_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.clone(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_clone_higher_order_tensor_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.clone(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_clone_method_higher_order_tensor_backward():
    a = np.random.rand(3, 4, 5, 2)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor.clone()
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_scalar_backward():
    a = np.array(4.0)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.select(a_tensor, ())
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_scalar_operator_backward():
    a = np.array(4.0)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[()]
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.ones_like(a)

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_vector_backward():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.select(a_tensor, slice(1, 4))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[1:4] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_vector_operator_backward():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[1:4]
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[1:4] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_matrix_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.select(a_tensor, (slice(1, 3), slice(0, 2)))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[1:3, 0:2] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_matrix_operator_backward():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[1:3, 0:2]
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[1:3, 0:2] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_tensor_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.select(a_tensor, (slice(None), slice(1, 3), slice(0, 2)))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[:, 1:3, 0:2] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_tensor_operator_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[:, 1:3, 0:2]
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[:, 1:3, 0:2] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_higher_order_tensor_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.select(
        a_tensor, (slice(1, 2), slice(None), slice(0, 3), slice(2, 4))
    )
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[1:2, :, 0:3, 2:4] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_higher_order_tensor_operator_backward():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[1:2, :, 0:3, 2:4]
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[1:2, :, 0:3, 2:4] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_ellipsis_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.select(a_tensor, (slice(None), ..., slice(1, 3)))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[:, :, 1:3] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_ellipsis_operator_backward():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[:, ..., 1:3]
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[:, :, 1:3] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_mixed_slices_backward():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.select(a_tensor, (slice(1, 3), ..., 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[1:3, :, 2] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_mixed_slices_operator_backward():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[1:3, ..., 2]
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[1:3, :, 2] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_single_dimension_backward():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.select(a_tensor, slice(1, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[1:2, :, :] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


def test_select_single_dimension_operator_backward():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[1:2]
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad_a = np.zeros_like(a)
    expected_grad_a[1:2, :, :] = 1

    assert a_tensor.grad is not None
    np.testing.assert_allclose(
        a_tensor.grad.data, expected_grad_a, rtol=1e-7, atol=1e-7
    )


# TODO tests for flatten and concat
