import numpy as np
import deepnet
import deepnet.nn.functional as f


def test_add_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.add(a_tensor, b_tensor)
    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    h = 1e-7
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

    h = 1e-7
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

    h = 1e-7
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

    h = 1e-7
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

    h = 1e-7
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

    h = 1e-7
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

    h = 1e-7
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

    h = 1e-7
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

    h = 1e-7
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

    h = 1e-7
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

    h = 1e-7
    expected_grad_a = ((a + h) / b - a / b) / h
    expected_grad_b = (a / (b + h) - a / b) / h
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

    h = 1e-7
    expected_grad_a = ((a + h) / b - a / b) / h
    expected_grad_b = (a / (b + h) - a / b) / h
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-5)


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

        print("All tests passed")


if __name__ == "__main__":
    main()
