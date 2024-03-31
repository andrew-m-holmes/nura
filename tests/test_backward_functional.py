import numpy as np
import nura
import nura.functional as f


def test_add_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.add(a_tensor, b_tensor)
    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected = (a + b + h - (a + b)) / h
    np.testing.assert_allclose(grad_a.data, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected, rtol=1e-5, atol=1e-5)


def test_add_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.add(a_tensor, b_tensor)

    v = nura.oneslike(result_tensor)
    result_tensor.backward(v)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected = (a + b + h - (a + b)) / h
    np.testing.assert_allclose(grad_a.data, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected, rtol=1e-5, atol=1e-5)


def test_add_backward_matrix():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.add(a_tensor, b_tensor)

    m = nura.oneslike(result_tensor)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected = (a + b + h - (a + b)) / h
    np.testing.assert_allclose(grad_a.data, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected, rtol=1e-5, atol=1e-5)


def test_sub_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.sub(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = (a + h - b - (a - b)) / h
    expected_grad_b = (a - (b + h) - (a - b)) / h
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_sub_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.sub(a_tensor, b_tensor)

    v = nura.oneslike(result_tensor)
    result_tensor.backward(v)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = (a + h - b - (a - b)) / h
    expected_grad_b = (a - (b + h) - (a - b)) / h
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_sub_backward_matrix():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.sub(a_tensor, b_tensor)

    m = nura.oneslike(result_tensor)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = (a + h - b - (a - b)) / h
    expected_grad_b = (a - (b + h) - (a - b)) / h
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_mul_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.mul(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) * b - a * b) / h
    expected_grad_b = (a * (b + h) - a * b) / h
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_mul_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.mul(a_tensor, b_tensor)

    v = nura.ones((4,), dtype=nura.float)
    result_tensor.backward(v)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) * b - a * b) / h
    expected_grad_b = (a * (b + h) - a * b) / h
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_mul_backward_matrix():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.mul(a_tensor, b_tensor)

    m = nura.ones((5, 5), dtype=nura.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) * b - a * b) / h
    expected_grad_b = (a * (b + h) - a * b) / h
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_div_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.div(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) / b - a / b) / h
    expected_grad_b = (a / (b + h) - a / b) / h
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_div_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.div(a_tensor, b_tensor)

    v = nura.ones((4,), dtype=nura.float)
    result_tensor.backward(v)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_grad_b = (a / (b + h) - a / (b - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_div_backward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.div(a_tensor, b_tensor)

    m = nura.ones((3, 3), dtype=nura.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_grad_b = (a / (b + h) - a / (b - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_dot_backward_vector_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.dot(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = b
    expected_grad_b = a
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_dot_backward_matrix_vector():
    a = np.random.rand(3, 5)
    b = np.random.rand(5)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.dot(a_tensor, b_tensor)

    ones = nura.oneslike(result_tensor)
    result_tensor.backward(ones)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.outer(ones.data, b)
    expected_grad_b = np.dot(a.T, ones.data)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_dot_backward_vector_matrix():
    a = np.random.rand(7)
    b = np.random.rand(7, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.dot(a_tensor, b_tensor)

    ones = nura.oneslike(result_tensor)
    result_tensor.backward(ones)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.dot(b.data, ones.data)
    expected_grad_b = np.outer(a.data, ones.data)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_dot_backward_matrix_matrix():
    a = np.random.rand(3, 4)
    b = np.random.rand(4, 2)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.dot(a_tensor, b_tensor)

    ones = nura.oneslike(result_tensor)
    result_tensor.backward(ones)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.dot(ones.data, b.T)
    expected_grad_b = np.dot(a.T, ones.data)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_matmul_backward_same_shape():
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)

    ones = np.ones((2, 2))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.matmul(ones, b.T)
    expected_grad_b = np.matmul(a.T, ones)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_matmul_backward_different_shape():
    a = np.random.rand(3, 2)
    b = np.random.rand(2, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)

    ones = np.ones((3, 4))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.matmul(ones, b.T)
    expected_grad_b = np.matmul(a.T, ones)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_matmul_backward_rank3_same_shape():
    a = np.random.rand(5, 5, 5)
    b = np.random.rand(5, 5, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)

    ones = np.ones((5, 5, 5))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.matmul(ones, b.transpose(0, 2, 1))
    expected_grad_b = np.matmul(a.transpose(0, 2, 1), ones)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_matmul_backward_rank3_different_shape():
    a = np.random.rand(3, 4, 5)
    b = np.random.rand(3, 5, 2)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)

    ones = np.ones((3, 4, 2))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.matmul(ones, b.transpose(0, 2, 1))
    expected_grad_b = np.matmul(a.transpose(0, 2, 1), ones)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_matmul_backward_different_ranks():
    a = np.random.rand(6, 2, 9, 4, 3)
    b = np.random.rand(3, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)

    ones = np.ones((6, 2, 9, 4, 4))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.matmul(ones, np.swapaxes(b.data, -2, -1))
    expected_grad_b = np.sum(
        np.matmul(np.swapaxes(a.data, -2, -1), ones), axis=(0, 1, 2)
    )
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_pow_backward_scalar():
    a = np.random.rand()
    b = 2.0

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward()

    grad_a, grad_b = a_tensor.grad, b_tensor.grad
    h = 1e-8
    expected_grad_a = (np.power(a + h, b) - np.power(a - h, b)) / (2 * h)
    expected_grad_b = (np.power(a, b + h) - np.power(a, b - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_pow_backward_vector():
    a = np.random.rand(5)
    b = np.array(3.0)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)

    v = nura.oneslike(result_tensor)
    result_tensor.backward(v)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = (np.power(a + h, b) - np.power(a - h, b)) / (2 * h)
    expected_grad_b = (np.power(a, b + h) - np.power(a, b - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, np.sum(expected_grad_b, axis=0), rtol=1e-5, atol=1e-5
    )


def test_pow_backward_matrix():
    a = np.random.rand(5, 5)
    b = np.array(4.0)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b, usegrad=True)
    result_tensor = f.pow(a_tensor, b_tensor)

    m = nura.oneslike(result_tensor)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = (np.power(a + h, b) - np.power(a - h, b)) / (2 * h)
    expected_grad_b = (np.power(a, b + h) - np.power(a, b - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, np.sum(expected_grad_b, axis=(0, 1)), rtol=1e-5, atol=1e-5
    )


def test_pow_backward_vector_exp():
    a = np.random.rand(4)
    b = np.full_like(a, 2)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    ones = np.ones(4)
    v = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(v)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.power(a + h, b) - np.power(a - h, b)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_pow_backward_matrix_exp():
    a = np.random.rand(3, 3)
    b = np.full_like(a, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    ones = np.ones((3, 3))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.power(a + h, b) - np.power(a - h, b)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_square_backward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.square(a + h) - np.square(a - h)) / (2 * h)
    np.testing.assert_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_square_backward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    ones = np.ones(5)
    v = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(v)

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.square(a + h) - np.square(a - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_square_backward_matrix():
    a = np.random.rand(5, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.square(a_tensor)
    ones = np.ones((5, 5))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.square(a + h) - np.square(a - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_sqrt_backward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sqrt(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.sqrt(a + h) - np.sqrt(a - h)) / (2 * h)
    np.testing.assert_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_sqrt_backward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sqrt(a_tensor)
    ones = np.ones(5)
    v = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(v)

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.sqrt(a + h) - np.sqrt(a - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_sqrt_backward_matrix():
    a = np.random.rand(5, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sqrt(a_tensor)
    ones = np.ones((5, 5))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.sqrt(a + h) - np.sqrt(a - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_exp_backward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.exp(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.exp(a + h) - np.exp(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_exp_backward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.exp(a_tensor)

    ones = np.ones(5)
    v = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(v)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.exp(a + h) - np.exp(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_exp_backward_matrix():
    a = np.random.rand(5, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.exp(a_tensor)

    ones = np.ones((5, 4))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.exp(a + h) - np.exp(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_log_backward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.log(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.log(a + h) - np.log(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_log_backward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.log(a_tensor)

    ones = np.ones(5)
    v = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(v)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.log(a + h) - np.log(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_log_backward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.log(a_tensor)

    ones = np.ones((3, 3))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.log(a + h) - np.log(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_sin_backward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sin(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.sin(a + h) - np.sin(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_sin_backward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sin(a_tensor)

    ones = np.ones(5)
    v = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(v)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.sin(a + h) - np.sin(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_sin_backward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.sin(a_tensor)

    ones = np.ones((3, 3))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.sin(a + h) - np.sin(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_cos_backward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.cos(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.cos(a + h) - np.cos(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_cos_backward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.cos(a_tensor)

    ones = np.ones(5)
    v = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(v)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.cos(a + h) - np.cos(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_cos_backward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.cos(a_tensor)

    ones = np.ones((3, 3))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.cos(a + h) - np.cos(a - h)) / (2 * h)
    np.testing.assert_allclose(grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_sum_backward_single_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.sum(a_tensor, 1)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_sum_backward_multiple_dims():
    a = np.random.rand(4, 5, 6)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.sum(a_tensor, (0, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_sum_backward_keepdims_true():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.sum(a_tensor, 1, keepdims=True)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_sum_backward_keepdims_false():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.sum(a_tensor, 1, keepdims=False)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_sum_backward_single_element_tensor():
    a = np.random.rand(1)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.sum(a_tensor, 0)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_sum_backward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.sum(a_tensor, (1, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_max_backward_single_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.max(a_tensor, 1)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_max_backward_multiple_dims():
    a = np.random.rand(4, 5, 6)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.max(a_tensor, (0, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_max_backward_keepdims_true():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.max(a_tensor, 1, keepdims=True)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_max_backward_keepdims_false():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.max(a_tensor, 1, keepdims=False)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_max_backward_single_element_tensor():
    a = np.random.rand(1)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.max(a_tensor, 0)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_max_backward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.max(a_tensor, (1, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_min_backward_single_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.min(a_tensor, 1)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_min_backward_multiple_dims():
    a = np.random.rand(4, 5, 6)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.min(a_tensor, (0, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_min_backward_keepdims_true():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.min(a_tensor, 1, keepdims=True)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_min_backward_keepdims_false():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.min(a_tensor, 1, keepdims=False)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_min_backward_single_element_tensor():
    a = np.random.rand(1)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.min(a_tensor, 0)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_min_backward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.min(a_tensor, (1, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_abs_backward_scalar():
    a = np.random.rand() * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.abs(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.absolute(a + h) - np.absolute(a - h)) / (2 * h)
    np.testing.assert_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_abs_backward_vector():
    a = np.random.rand(5) * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.abs(a_tensor)
    ones = np.ones(5)
    v = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(v)

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.absolute(a + h) - np.absolute(a - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_abs_backward_matrix():
    a = np.random.rand(3, 3) * np.random.choice([-1, 1], size=(3, 3))

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.abs(a_tensor)
    ones = np.ones((3, 3))
    m = nura.tensor(ones, dtype=nura.float)
    result_tensor.backward(m)

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.absolute(a + h) - np.absolute(a - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_pos_backward_scalar():
    a = np.random.rand() * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pos(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.positive(a + h) - np.positive(a - h)) / (2 * h)
    np.testing.assert_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_pos_backward_vector():
    a = np.random.rand(5) * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pos(a_tensor)
    v = nura.oneslike(result_tensor)
    result_tensor.backward(v)

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.positive(a + h) - np.positive(a - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_pos_backward_matrix():
    a = np.random.rand(3, 3) * np.random.choice([-1, 1], size=(3, 3))

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.pos(a_tensor)
    m = nura.oneslike(result_tensor)
    result_tensor.backward(m)

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.positive(a + h) - np.positive(a - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_neg_backward_scalar():
    a = np.random.rand() * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.neg(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.negative(a + h) - np.negative(a - h)) / (2 * h)
    np.testing.assert_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_neg_backward_vector():
    a = np.random.rand(5) * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.neg(a_tensor)
    v = nura.oneslike(result_tensor)
    result_tensor.backward(v)

    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.negative(a + h) - np.negative(a - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_neg_backward_matrix():
    a = np.random.rand(3, 3) * np.random.choice([-1, 1], size=(3, 3))

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = f.neg(a_tensor)
    m = nura.oneslike(result_tensor)
    result_tensor.backward(m)
    grad_a = a_tensor.grad
    h = 1e-8
    expected_grad_a = (np.negative(a + h) - np.negative(a - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad_a.data, expected_grad_a, decimal=5)


def test_squeeze_backward_rank1_v0():
    a = np.random.rand(1)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.squeeze(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_squeeze_backward_rank1_v1():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.squeeze(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_squeeze_backward_rank2_v0():
    a = np.random.rand(5, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.squeeze(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_squeeze_backward_rank2_v1():
    a = np.random.rand(3, 1)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.squeeze(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_squeeze_backward_multi_v0():
    a = np.random.rand(3, 1, 5, 2, 1, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.squeeze(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_squeeze_backward_multi_v1():
    a = np.random.rand(1, 1, 1, 1, 1, 1, 1, 69, 1)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.squeeze(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_squeeze_backward_multi_v2():
    a = np.random.rand(4, 4, 5, 6, 2)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.squeeze(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_unsqueeze_backward_multi_v0():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.unsqueeze(a_tensor, (0, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_unsqueeze_backward_multi_v1():
    a = np.random.rand(2, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.unsqueeze(a_tensor, (1, 3, 4))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_unsqueeze_backward_multi_v2():
    a = np.random.rand(5, 6, 7, 8)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.unsqueeze(a_tensor, (0, 2, 5))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_unsqueeze_backward_multi_v3():
    a = np.random.rand(4, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.unsqueeze(a_tensor, (1,))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_unsqueeze_backward_multi_v4():
    a = np.random.rand(2, 5, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.unsqueeze(a_tensor, (0, 3))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_transpose_backward_rank2_v0():
    a = np.random.rand(5, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.transpose(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_transpose_backward_rank2_v1():
    a = np.random.rand(3, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.transpose(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_transpose_backward_rank3_v0():
    a = np.random.rand(4, 3, 2)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.transpose(a_tensor, 1, 2)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_transpose_backward_multi_v0():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.transpose(a_tensor, -2, -3)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_transpose_backward_multi_v1():
    a = np.random.rand(3, 4, 5, 6)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.transpose(a_tensor, 0, 3)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_permute_backward_rank2_v0():
    a = np.random.rand(10, 20)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.permute(a_tensor, (1, 0))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_permute_backward_rank3_v0():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.permute(a_tensor, (1, 0, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_permute_backward_rank3_v1():
    a = np.random.rand(64, 10, 512)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.permute(a_tensor, (2, 1, 0))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_permute_backward_rank4_v0():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.permute(a_tensor, (3, 2, 1, 0))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_permute_backward_rank4_v1():
    a = np.random.rand(5, 6, 7, 8)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.permute(a_tensor, (0, 3, 2, 1))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_view_backward_rank1_to_rank2():
    a = np.random.rand(12)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.view(a_tensor, (4, 3))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_view_backward_rank2_to_rank1():
    a = np.random.rand(5, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.view(a_tensor, (20,))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_view_backward_rank2_to_rank3():
    a = np.random.rand(8, 6)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.view(a_tensor, (2, 4, 6))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_view_backward_rank3_to_rank2():
    a = np.random.rand(3, 5, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.view(a_tensor, (15, 4))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_view_backward_rank3_to_rank4():
    a = np.random.rand(3, 4, 2)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.view(a_tensor, (3, 2, 2, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_view_backward_rank4_to_rank2():
    a = np.random.rand(3, 2, 4, 2)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.view(a_tensor, (6, 8))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_view_backward_with_negative_dim():
    a = np.random.rand(4, 3, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.view(a_tensor, (-1, 5))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_reshape_backward_rank1_to_rank2():
    a = np.random.rand(10)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.reshape(a_tensor, (5, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_reshape_backward_rank2_to_rank1():
    a = np.random.rand(4, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.reshape(a_tensor, (12,))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_reshape_backward_rank2_to_rank3():
    a = np.random.rand(6, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.reshape(a_tensor, (2, 3, 4))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_reshape_backward_rank3_to_rank2():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.reshape(a_tensor, (6, 4))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_reshape_backward_rank3_to_rank4():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.reshape(a_tensor, (2, 2, 3, 2))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_reshape_backward_rank4_to_rank2():
    a = np.random.rand(2, 2, 3, 2)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.reshape(a_tensor, (4, 6))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_reshape_backward_with_negative_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.reshape(a_tensor, (-1, 5))
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim == a.shape


def test_clone_backward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.clone(a_tensor)
    result_tensor.backward(nura.tensor(1.0))

    grad_a = a_tensor.grad
    assert np.allclose(grad_a.data, 1.0)


def test_clone_backward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.clone(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert np.allclose(grad_a.data, np.ones_like(a))


def test_clone_backward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.clone(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert np.allclose(grad_a.data, np.ones_like(a))


def test_clone_backward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = nura.clone(a_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))

    grad_a = a_tensor.grad
    assert np.allclose(grad_a.data, np.ones_like(a))


def test_slice_backward_single_index():
    a = np.random.rand(5, 5)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[2, :]
    result_tensor.backward(nura.oneslike(result_tensor))

    expected_grad = np.zeros_like(a)
    expected_grad[2, :] = 1
    np.testing.assert_array_almost_equal(a_tensor.grad.data, expected_grad, decimal=5)


def test_slice_backward_range():
    a = np.random.rand(10, 10)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[2:5, 3:7]
    gradient_mask = nura.oneslike(result_tensor)
    result_tensor.backward(gradient_mask)

    expected_grad = np.zeros_like(a)
    expected_grad[2:5, 3:7] = 1
    np.testing.assert_array_almost_equal(a_tensor.grad.data, expected_grad, decimal=5)


def test_slice_backward_step():
    a = np.random.rand(8, 8)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[::2, ::3]
    gradient_mask = nura.oneslike(result_tensor)
    result_tensor.backward(gradient_mask)

    expected_grad = np.zeros_like(a)
    expected_grad[::2, ::3] = 1
    np.testing.assert_array_almost_equal(a_tensor.grad.data, expected_grad, decimal=5)


def test_slice_backward_negative_indices():
    a = np.random.rand(6, 6)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[-3:, -3:]
    gradient_mask = nura.oneslike(result_tensor)
    result_tensor.backward(gradient_mask)

    expected_grad = np.zeros_like(a)
    expected_grad[-3:, -3:] = 1
    np.testing.assert_array_almost_equal(a_tensor.grad.data, expected_grad, decimal=5)


def test_slice_backward_mixed_indices():
    a = np.random.rand(7, 7)

    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = a_tensor[1:5, -3]
    gradient_mask = nura.oneslike(result_tensor)
    result_tensor.backward(gradient_mask)

    expected_grad = np.zeros_like(a)
    expected_grad[1:5, -3] = 1
    np.testing.assert_array_almost_equal(a_tensor.grad.data, expected_grad, decimal=5)
