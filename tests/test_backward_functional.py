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
