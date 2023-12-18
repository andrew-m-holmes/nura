import numpy as np
import deepnet
import deepnet.functional as f


def test_add_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.add(a_tensor, b_tensor)
    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected = (a + b + h - (a + b)) / h
    np.testing.assert_allclose(
        grad_a.data, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected, rtol=1e-5, atol=1e-5)


def test_add_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.add(a_tensor, b_tensor)

    v = deepnet.ones((4,), dtype=deepnet.float)
    result_tensor.backward(v)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected = (a + b + h - (a + b)) / h
    np.testing.assert_allclose(
        grad_a.data, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected, rtol=1e-5, atol=1e-5)


def test_add_backward_matrix():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.add(a_tensor, b_tensor)

    m = deepnet.ones((5, 5), dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected = (a + b + h - (a + b)) / h
    np.testing.assert_allclose(
        grad_a.data, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected, rtol=1e-5, atol=1e-5)


def test_sub_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.sub(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = (a + h - b - (a - b)) / h
    expected_grad_b = (a - (b + h) - (a - b)) / h
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_sub_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.sub(a_tensor, b_tensor)

    v = deepnet.ones((4), dtype=deepnet.float)
    result_tensor.backward(v)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = (a + h - b - (a - b)) / h
    expected_grad_b = (a - (b + h) - (a - b)) / h
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_sub_backward_matrix():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.sub(a_tensor, b_tensor)

    m = deepnet.ones((5, 5), dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = (a + h - b - (a - b)) / h
    expected_grad_b = (a - (b + h) - (a - b)) / h
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_mul_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.mul(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) * b - a * b) / h
    expected_grad_b = (a * (b + h) - a * b) / h
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_mul_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.mul(a_tensor, b_tensor)

    v = deepnet.ones((4,), dtype=deepnet.float)
    result_tensor.backward(v)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) * b - a * b) / h
    expected_grad_b = (a * (b + h) - a * b) / h
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_mul_backward_matrix():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.mul(a_tensor, b_tensor)

    m = deepnet.ones((5, 5), dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) * b - a * b) / h
    expected_grad_b = (a * (b + h) - a * b) / h
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_div_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.div(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) / b - a / b) / h
    expected_grad_b = (a / (b + h) - a / b) / h
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_div_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.div(a_tensor, b_tensor)

    v = deepnet.ones((4,), dtype=deepnet.float)
    result_tensor.backward(v)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_grad_b = (a / (b + h) - a / (b - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_div_backward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.div(a_tensor, b_tensor)

    m = deepnet.ones((3, 3), dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_grad_b = (a / (b + h) - a / (b - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)

# Using symbolic differentiaton as numeric differentiation, gives unwanted results


def test_matmul_backward_same_shape():
    a = np.random.rand(2, 2)
    b = np.random.rand(2, 2)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)

    ones = np.ones((2, 2))
    m = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.matmul(ones, b.T)
    expected_grad_b = np.matmul(a.T, ones)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_matmul_backward_different_shape():
    a = np.random.rand(3, 2)
    b = np.random.rand(2, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)

    ones = np.ones((3, 4))
    m = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.matmul(ones, b.T)
    expected_grad_b = np.matmul(a.T, ones)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_matmul_backward_rank3_same_shape():
    a = np.random.rand(5, 5, 5)
    b = np.random.rand(5, 5, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)

    ones = np.ones((5, 5, 5))
    m = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.matmul(ones, b.transpose(0, 2, 1))
    expected_grad_b = np.matmul(a.transpose(0, 2, 1), ones)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_matmul_backward_rank3_different_shape():
    a = np.random.rand(3, 4, 5)
    b = np.random.rand(3, 5, 2)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.matmul(a_tensor, b_tensor)

    ones = np.ones((3, 4, 2))
    m = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    expected_grad_a = np.matmul(ones, b.transpose(0, 2, 1))
    expected_grad_b = np.matmul(a.transpose(0, 2, 1), ones)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_pow_backward_scalar():
    a = np.random.rand()
    b = 2.

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.pow(a_tensor, b_tensor)
    result_tensor.backward()

    grad_a, grad_b = a_tensor.grad, b_tensor.grad
    h = 1e-8
    expected_grad_a = (
        np.power(a + h, b) - np.power(a - h, b)) / (2 * h)
    expected_grad_b = (
        np.power(a, b + h) - np.power(a, b - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


def test_pow_backward_vector():
    a = np.random.rand(5)
    b = 3.

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.pow(a_tensor, b_tensor)

    ones = np.ones(5)
    v = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(v)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = (
        np.power(a + h, b) - np.power(a - h, b)) / (2 * h)
    expected_grad_b = (
        np.power(a, b + h) - np.power(a, b - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, np.sum(expected_grad_b, axis=0),
        rtol=1e-5, atol=1e-5)


def test_pow_backward_matrix():
    a = np.random.rand(5, 5)
    b = 4.

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.pow(a_tensor, b_tensor)

    ones = np.ones((5, 5))
    m = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-8
    expected_grad_a = (
        np.power(a + h, b) - np.power(a - h, b)) / (2 * h)
    expected_grad_b = (
        np.power(a, b + h) - np.power(a, b - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, np.sum(expected_grad_b, axis=(0, 1)),
        rtol=1e-5, atol=1e-5)


def test_pow_backward_vector_exp():
    a = np.random.rand(4)
    b = np.full_like(a, 2)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    ones = np.ones(4)
    v = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(v)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (
        np.power(a + h, b) - np.power(a - h, b)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_pow_backward_matrix_exp():
    a = np.random.rand(3, 3)
    b = np.full_like(a, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    ones = np.ones((3, 3))
    m = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (
        np.power(a + h, b) - np.power(a - h, b)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_sine_backward_scalar():
    a = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = f.sine(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.sin(a + h) - np.sin(a - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_sine_backward_vector():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = f.sine(a_tensor)

    ones = np.ones(5)
    v = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(v)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.sin(a + h) - np.sin(a - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_sine_backward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = f.sine(a_tensor)

    ones = np.ones((3, 3))
    m = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.sin(a + h) - np.sin(a - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_cosine_backward_scalar():
    a = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = f.cosine(a_tensor)
    result_tensor.backward()

    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.cos(a + h) - np.cos(a - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_cosine_backward_vector():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = f.cosine(a_tensor)

    ones = np.ones(5)
    v = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(v)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.cos(a + h) - np.cos(a - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)


def test_cosine_backward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = f.cosine(a_tensor)

    ones = np.ones((3, 3))
    m = deepnet.tensor(ones, dtype=deepnet.float)
    result_tensor.backward(m)
    grad_a = a_tensor.grad

    h = 1e-8
    expected_grad_a = (np.cos(a + h) - np.cos(a - h)) / (2 * h)
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)

def test_sum_backward_single_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.sum(a_tensor, 1)
    print(result_tensor.dim())
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_sum_backward_multiple_dims():
    a = np.random.rand(4, 5, 6)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.sum(a_tensor, (0, 2))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_sum_backward_keepdims_true():
    a = np.random.rand(2, 3, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.sum(a_tensor, 1, keepdims=True)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_sum_backward_keepdims_false():
    a = np.random.rand(2, 3, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.sum(a_tensor, 1, keepdims=False)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_sum_backward_single_element_tensor():
    a = np.random.rand(1)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.sum(a_tensor, 0)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_sum_backward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.sum(a_tensor, (1, 2))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_squeeze_backward_rank1_v0():
    a = np.random.rand(1)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.squeeze(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_squeeze_backward_rank1_v1():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.squeeze(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_squeeze_backward_rank2_v0():
    a = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.squeeze(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_squeeze_backward_rank2_v1():
    a = np.random.rand(3, 1)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.squeeze(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_squeeze_backward_multi_v0():
    a = np.random.rand(3, 1, 5, 2, 1, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.squeeze(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_squeeze_backward_multi_v1():
    a = np.random.rand(1, 1, 1, 1, 1, 1, 1, 69, 1)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.squeeze(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_squeeze_backward_multi_v2():
    a = np.random.rand(4, 4, 5, 6, 2)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.squeeze(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_unsqueeze_backward_multi_v0():
    a = np.random.rand(3, 4, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.unsqueeze(a_tensor, (0, 2))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_unsqueeze_backward_multi_v1():
    a = np.random.rand(2, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.unsqueeze(a_tensor, (1, 3, 4))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_unsqueeze_backward_multi_v2():
    a = np.random.rand(5, 6, 7, 8)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.unsqueeze(a_tensor, (0, 2, 5))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_unsqueeze_backward_multi_v3():
    a = np.random.rand(4, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.unsqueeze(a_tensor, (1,))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_unsqueeze_backward_multi_v4():
    a = np.random.rand(2, 5, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.unsqueeze(a_tensor, (0, 3))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_transpose_backward_rank2_v0():
    a = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.transpose(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_transpose_backward_rank2_v1():
    a = np.random.rand(3, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.transpose(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_transpose_backward_rank3_v0():
    a = np.random.rand(4, 3, 2)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.transpose(a_tensor, 1, 2)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_transpose_backward_multi_v0():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.transpose(a_tensor, -2, -3)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_transpose_backward_multi_v1():
    a = np.random.rand(3, 4, 5, 6)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.transpose(a_tensor, 0, 3)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_permute_backward_rank2_v0():
    a = np.random.rand(10, 20)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.permute(a_tensor, (1, 0))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_permute_backward_rank3_v0():
    a = np.random.rand(3, 4, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.permute(a_tensor, (1, 0, 2))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_permute_backward_rank3_v1():
    a = np.random.rand(64, 10, 512)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.permute(a_tensor, (2, 1, 0))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_permute_backward_rank4_v0():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.permute(a_tensor, (3, 2, 1, 0))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_permute_backward_rank4_v1():
    a = np.random.rand(5, 6, 7, 8)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.permute(a_tensor, (0, 3, 2, 1))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_view_backward_rank1_to_rank2():
    a = np.random.rand(12)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.view(a_tensor, (4, 3))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_view_backward_rank2_to_rank1():
    a = np.random.rand(5, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.view(a_tensor, (20,))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_view_backward_rank2_to_rank3():
    a = np.random.rand(8, 6)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.view(a_tensor, (2, 4, 6))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_view_backward_rank3_to_rank2():
    a = np.random.rand(3, 5, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.view(a_tensor, (15, 4))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_view_backward_rank3_to_rank4():
    a = np.random.rand(3, 4, 2)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.view(a_tensor, (3, 2, 2, 2))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_view_backward_rank4_to_rank2():
    a = np.random.rand(3, 2, 4, 2)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.view(a_tensor, (6, 8))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_view_backward_with_negative_dim():
    a = np.random.rand(4, 3, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.view(a_tensor, (-1, 5))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape



def test_reshape_backward_rank1_to_rank2():
    a = np.random.rand(10)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.reshape(a_tensor, (5, 2))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_reshape_backward_rank2_to_rank1():
    a = np.random.rand(4, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.reshape(a_tensor, (12,))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_reshape_backward_rank2_to_rank3():
    a = np.random.rand(6, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.reshape(a_tensor, (2, 3, 4))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_reshape_backward_rank3_to_rank2():
    a = np.random.rand(2, 3, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.reshape(a_tensor, (6, 4))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_reshape_backward_rank3_to_rank4():
    a = np.random.rand(2, 3, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.reshape(a_tensor, (2, 2, 3, 2))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_reshape_backward_rank4_to_rank2():
    a = np.random.rand(2, 2, 3, 2)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.reshape(a_tensor, (4, 6))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape


def test_reshape_backward_with_negative_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.reshape(a_tensor, (-1, 5))
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert grad_a.dim() == a.shape

def test_clone_backward_scalar():
    a = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.clone(a_tensor)
    result_tensor.backward(deepnet.tensor(1.0))

    grad_a = a_tensor.grad
    assert np.allclose(grad_a.data, 1.0)

def test_clone_backward_vector():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.clone(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert np.allclose(grad_a.data, np.ones_like(a))

def test_clone_backward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.clone(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert np.allclose(grad_a.data, np.ones_like(a))

def test_clone_backward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    result_tensor = deepnet.clone(a_tensor)
    result_tensor.backward(deepnet.ones_like(result_tensor))

    grad_a = a_tensor.grad
    assert np.allclose(grad_a.data, np.ones_like(a))


def main():

    with deepnet.use_grad():

        # Add Backward Tests

        test_add_backward_scalar()
        test_add_backward_vector()
        test_add_backward_matrix()

        # Sub Backward Tests

        test_sub_backward_scalar()
        test_sub_backward_vector()
        test_sub_backward_matrix()

        # Mul Backward Tests

        test_mul_backward_scalar()
        test_mul_backward_vector()
        test_sub_backward_matrix()

        # Div Backward Tests

        test_div_backward_scalar()
        test_div_backward_vector()
        test_div_backward_matrix()

        # Matmul Backward Tests

        test_matmul_backward_same_shape()
        test_matmul_backward_different_shape()
        test_matmul_backward_rank3_same_shape()
        test_matmul_backward_rank3_different_shape()

        # Sine Backward Tests

        test_sine_backward_scalar()
        test_sine_backward_vector()
        test_sine_backward_matrix()

        # Cosine Backward Tests

        test_cosine_backward_scalar()
        test_cosine_backward_vector()
        test_cosine_backward_matrix()

        # Pow Backward Tests

        test_pow_backward_scalar()
        test_pow_backward_vector()
        test_pow_backward_matrix()

        # Pow Backward (exponent is not a scalar)

        test_pow_backward_vector_exp()
        test_pow_backward_matrix_exp()

        # Sum Backward Tests

        test_sum_backward_single_dim()
        test_sum_backward_multiple_dims()
        test_sum_backward_higher_rank_tensor()

        test_sum_backward_keepdims_false()
        test_sum_backward_keepdims_true()
        test_sum_backward_single_element_tensor()
    

        # Squeeze Backward Tests

        test_squeeze_backward_rank1_v0()
        test_squeeze_backward_rank1_v1()
        test_squeeze_backward_rank2_v0()
        test_squeeze_backward_rank2_v1()

        test_squeeze_backward_multi_v0()
        test_squeeze_backward_multi_v1()
        test_squeeze_backward_multi_v2()

        # Unsqueeze Backward Tests

        test_unsqueeze_backward_multi_v0()
        test_unsqueeze_backward_multi_v1()
        test_unsqueeze_backward_multi_v2()

        test_unsqueeze_backward_multi_v3()
        test_unsqueeze_backward_multi_v4()

        # Transpose Backward Tests

        test_transpose_backward_rank2_v0()
        test_transpose_backward_rank2_v1()
        test_transpose_backward_rank3_v0()

        test_transpose_backward_multi_v0()
        test_transpose_backward_multi_v1()

        # Permute Backward Tests

        test_permute_backward_rank2_v0()
        test_permute_backward_rank3_v0()
        test_permute_backward_rank3_v1()
        test_permute_backward_rank4_v0()
        test_permute_backward_rank4_v1()

        # View Backward Tests

        test_view_backward_rank1_to_rank2()
        test_view_backward_rank2_to_rank3()
        test_view_backward_rank3_to_rank4()

        test_view_backward_rank2_to_rank1()
        test_view_backward_rank3_to_rank2()
        test_view_backward_rank4_to_rank2()
        test_view_backward_with_negative_dim()

        # Reshape Backward Tests

        test_reshape_backward_rank1_to_rank2()
        test_reshape_backward_rank2_to_rank3()
        test_reshape_backward_rank3_to_rank4()

        test_reshape_backward_rank2_to_rank1()
        test_reshape_backward_rank3_to_rank2()
        test_reshape_backward_rank4_to_rank2()
        test_reshape_backward_with_negative_dim()

        # Clone Backward Tests

        test_clone_backward_scalar()
        test_clone_backward_vector()
        test_clone_backward_matrix()
        test_clone_backward_higher_rank_tensor()

        print("All tests passed")


if __name__ == "__main__":
    main()
