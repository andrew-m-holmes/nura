import numpy as np
import nura
import nura.nn.functional as f


def test_sigmoid_backward_scalar():
    x = np.array(0.5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.sigmoid(x_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return 1 / (1 + np.exp(-x))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_sigmoid_backward_vector():
    x = np.random.rand(5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.sigmoid(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 1 / (1 + np.exp(-x))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_sigmoid_backward_matrix():
    x = np.random.rand(3, 4)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.sigmoid(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 1 / (1 + np.exp(-x))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_sigmoid_backward_tensor():
    x = np.random.rand(2, 3, 4)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.sigmoid(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 1 / (1 + np.exp(-x))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_sigmoid_backward_higher_order_tensor():
    x = np.random.rand(2, 3, 4, 5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.sigmoid(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 1 / (1 + np.exp(-x))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_tanh_backward_scalar():
    x = np.array(0.7)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.tanh(x_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.tanh(x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_tanh_backward_vector():
    x = np.random.randn(5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.tanh(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.tanh(x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_tanh_backward_matrix():
    x = np.random.randn(3, 4)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.tanh(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.tanh(x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_tanh_backward_tensor():
    x = np.random.randn(2, 3, 4)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.tanh(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.tanh(x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_tanh_backward_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.tanh(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.tanh(x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_relu_backward_scalar():
    x = np.array(-0.5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.relu(x_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.maximum(0, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_relu_backward_vector():
    x = np.random.randn(6)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.relu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.maximum(0, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_relu_backward_matrix():
    x = np.random.randn(4, 5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.relu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.maximum(0, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_relu_backward_tensor():
    x = np.random.randn(3, 4, 2)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.relu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.maximum(0, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_relu_backward_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.relu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.maximum(0, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_relu6_backward_scalar():
    x = np.array(3.0)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.relu6(x_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.minimum(np.maximum(0, x), 6)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_relu6_backward_vector():
    x = np.random.randn(6) * 5
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.relu6(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.minimum(np.maximum(0, x), 6)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_relu6_backward_matrix():
    x = np.random.randn(4, 3) * 5
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.relu6(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.minimum(np.maximum(0, x), 6)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_relu6_backward_tensor():
    x = np.random.randn(3, 4, 2) * 5
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.relu6(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.minimum(np.maximum(0, x), 6)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_relu6_backward_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 5) * 5
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.relu6(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.minimum(np.maximum(0, x), 6)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_leakyrelu_backward_scalar():
    x = np.array(-0.5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.leakyrelu(x_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.maximum(0.01 * x, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_leakyrelu_backward_vector():
    x = np.random.randn(6)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.leakyrelu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.maximum(0.01 * x, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_leakyrelu_backward_matrix():
    x = np.random.randn(4, 3)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.leakyrelu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.maximum(0.01 * x, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_leakyrelu_backward_tensor():
    x = np.random.randn(3, 4, 2)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.leakyrelu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.maximum(0.01 * x, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_leakyrelu_backward_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.leakyrelu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.maximum(0.01 * x, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_leakyrelu_backward_custom_alpha():
    x = np.random.randn(4, 3)
    alpha = 0.2
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.leakyrelu(x_tensor, alpha=alpha)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.maximum(alpha * x, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_leakyrelu_backward_custom_alpha_high_order():
    x = np.random.randn(2, 3, 4, 5)
    alpha = 0.05
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.leakyrelu(x_tensor, alpha=alpha)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.maximum(alpha * x, x)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_elu_backward_scalar():
    x = np.array(-0.5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.elu(x_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, np.exp(x) - 1)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_elu_backward_vector():
    x = np.random.randn(6)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.elu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, np.exp(x) - 1)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_elu_backward_matrix():
    x = np.random.randn(4, 3)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.elu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, np.exp(x) - 1)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_elu_backward_tensor():
    x = np.random.randn(3, 4, 2)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.elu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, np.exp(x) - 1)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_elu_backward_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.elu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, np.exp(x) - 1)

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_elu_backward_custom_alpha():
    x = np.random.randn(4, 3)
    alpha = 1.0
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.elu(x_tensor, alpha=alpha)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_elu_backward_custom_alpha_high_order():
    x = np.random.randn(2, 3, 4, 5)
    alpha = 0.5
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.elu(x_tensor, alpha=alpha)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_gelu_backward_scalar():
    x = np.array(0.5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.gelu(x_tensor)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_gelu_backward_vector():
    x = np.random.randn(6)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.gelu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_gelu_backward_matrix():
    x = np.random.randn(4, 3)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.gelu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_gelu_backward_tensor():
    x = np.random.randn(3, 4, 2)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.gelu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_gelu_backward_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 5)
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.gelu(x_tensor)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_celu_backward_scalar():
    x = np.array(-0.5)
    alpha = 1.0
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.celu(x_tensor, alpha=alpha)
    result_tensor.backward()
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_celu_backward_vector():
    x = np.random.randn(6)
    alpha = 1.0
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.celu(x_tensor, alpha=alpha)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_celu_backward_matrix():
    x = np.random.randn(4, 3)
    alpha = 1.0
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.celu(x_tensor, alpha=alpha)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_celu_backward_tensor():
    x = np.random.randn(3, 4, 2)
    alpha = 1.0
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.celu(x_tensor, alpha=alpha)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_celu_backward_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 5)
    alpha = 1.0
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.celu(x_tensor, alpha=alpha)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_celu_backward_custom_alpha():
    x = np.random.randn(4, 3)
    alpha = 0.75
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.celu(x_tensor, alpha=alpha)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)


def test_celu_backward_custom_alpha_high_order():
    x = np.random.randn(2, 3, 4, 5)
    alpha = 0.25
    x_tensor = nura.tensor(x, usegrad=True)
    result_tensor = f.celu(x_tensor, alpha=alpha)
    result_tensor.backward(nura.oneslike(result_tensor))
    h = 1e-7

    def func(x):
        return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))

    expected_grad = (func(x + h) - func(x - h)) / (2 * h)

    assert x_tensor.grad is not None
    np.testing.assert_allclose(x_tensor.grad.data, expected_grad, rtol=1e-7, atol=1e-7)
