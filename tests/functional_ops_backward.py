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

    nudge = 1e-10
    expected = (a + b + nudge - (a + b)) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected, rtol=1e-5, atol=1e-8)


def test_add_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.add(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected = (a + b + nudge - (a + b)) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected, rtol=1e-5, atol=1e-8)


def test_add_backward_matrix():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.add(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected = (a + b + nudge - (a + b)) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected, rtol=1e-5, atol=1e-8)


def test_sub_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.sub(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected_grad_a = (a + nudge - b - (a - b)) / nudge
    expected_grad_b = (a - (b + nudge) - (a - b)) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-8)


def test_sub_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.sub(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected_grad_a = (a + nudge - b - (a - b)) / nudge
    expected_grad_b = (a - (b + nudge) - (a - b)) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-8)


def test_sub_backward_matrix():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.sub(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected_grad_a = (a + nudge - b - (a - b)) / nudge
    expected_grad_b = (a - (b + nudge) - (a - b)) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-8)


def test_mul_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.mul(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected_grad_a = ((a + nudge) * b - a * b) / nudge
    expected_grad_b = (a * (b + nudge) - a * b) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-8)


def test_mul_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.mul(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected_grad_a = ((a + nudge) * b - a * b) / nudge
    expected_grad_b = (a * (b + nudge) - a * b) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-8)


def test_mul_backward_matrix():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.mul(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected_grad_a = ((a + nudge) * b - a * b) / nudge
    expected_grad_b = (a * (b + nudge) - a * b) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-8)


def test_div_backward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.div(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected_grad_a = ((a + nudge) / b - a / b) / nudge
    expected_grad_b = (a / (b + nudge) - a / b) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-8)


def test_div_backward_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.div(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected_grad_a = ((a + nudge) / b - a / b) / nudge
    expected_grad_b = (a / (b + nudge) - a / b) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-8)


def test_div_backward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a, use_grad=True)
    b_tensor = deepnet.tensor(b, use_grad=True)
    result_tensor = f.div(a_tensor, b_tensor)

    result_tensor.backward()
    grad_a, grad_b = a_tensor.grad, b_tensor.grad

    nudge = 1e-10
    expected_grad_a = ((a + nudge) / b - a / b) / nudge
    expected_grad_b = (a / (b + nudge) - a / b) / nudge
    np.testing.assert_allclose(
        grad_a.data, expected_grad_a, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        grad_b.data, expected_grad_b, rtol=1e-5, atol=1e-8)


def main():

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

    print("All tests passed")


if __name__ == "__main__":
    main()
