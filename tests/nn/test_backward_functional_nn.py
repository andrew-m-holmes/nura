import nura
import nura.nn.functional as f
import numpy as np


def test_relu_backward_scalar():
    def relu(z):
        return np.maximum(0, z)

    z = np.random.randn()
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.relu(z_tensor)
    result_tensor.backward()

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (relu(z + h) - relu(z - h)) / (2 * h)
    np.testing.assert_almost_equal(grad.data, expected_grad, decimal=5)


def test_relu_backward_vector():
    def relu(z):
        return np.maximum(0, z)

    z = np.random.randn(5)
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.relu(z_tensor)
    result_tensor.backward(nura.tensor(np.ones_like(z)))

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (relu(z + h) - relu(z - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_relu_backward_matrix():
    def relu(z):
        return np.maximum(0, z)

    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.relu(z_tensor)
    result_tensor.backward(nura.tensor(np.ones_like(z)))

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (relu(z + h) - relu(z - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_relu6_backward_scalar():
    def relu6(z):
        return np.clip(z, 0, 6)

    z = np.random.randn()
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.relu6(z_tensor)
    result_tensor.backward()

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (relu6(z + h) - relu6(z - h)) / (2 * h)
    np.testing.assert_almost_equal(grad.data, expected_grad, decimal=5)


def test_relu6_backward_vector():
    def relu6(z):
        return np.clip(z, 0, 6)

    z = np.random.randn(5)
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.relu6(z_tensor)
    result_tensor.backward(nura.tensor(np.ones_like(z)))

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (relu6(z + h) - relu6(z - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_relu6_backward_matrix():
    def relu6(z):
        return np.clip(z, 0, 6)

    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.relu6(z_tensor)
    result_tensor.backward(nura.tensor(np.ones_like(z)))

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (relu6(z + h) - relu6(z - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_leakyrelu_backward_scalar():
    def leakyrelu(z, slope=0.01):
        return np.where(z > 0, z, slope * z)

    z = np.random.randn()
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.leakyrelu(z_tensor)
    result_tensor.backward()

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (leakyrelu(z + h) - leakyrelu(z - h)) / (2 * h)
    np.testing.assert_almost_equal(grad.data, expected_grad, decimal=5)


def test_leakyrelu_backward_vector():
    def leakyrelu(z, slope=0.01):
        return np.where(z > 0, z, slope * z)

    z = np.random.randn(5)
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.leakyrelu(z_tensor)
    result_tensor.backward(nura.tensor(np.ones_like(z)))

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (leakyrelu(z + h) - leakyrelu(z - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_leakyrelu_backward_matrix():
    def leakyrelu(z, slope=0.01):
        return np.where(z > 0, z, slope * z)

    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.leakyrelu(z_tensor)
    result_tensor.backward(nura.tensor(np.ones_like(z)))

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (leakyrelu(z + h) - leakyrelu(z - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_leakyrelu_backward_custom_slope():
    def leakyrelu(z, slope=0.05):
        return np.where(z > 0, z, slope * z)

    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.leakyrelu(z_tensor, slope=0.05)
    result_tensor.backward(nura.tensor(np.ones_like(z)))

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (leakyrelu(z + h, slope=0.05) - leakyrelu(z - h, slope=0.05)) / (
        2 * h
    )
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_elu_backward_scalar():
    def elu(z, alpha=1.0):
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))

    z = np.random.randn()
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.elu(z_tensor)
    result_tensor.backward()

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (elu(z + h) - elu(z - h)) / (2 * h)
    np.testing.assert_almost_equal(grad.data, expected_grad, decimal=5)


def test_elu_backward_vector():
    def elu(z, alpha=1.0):
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))

    z = np.random.randn(5)
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.elu(z_tensor)
    result_tensor.backward(nura.tensor(np.ones_like(z)))

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (elu(z + h) - elu(z - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_elu_backward_matrix():
    def elu(z, alpha=1.0):
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))

    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.elu(z_tensor)
    result_tensor.backward(nura.tensor(np.ones_like(z)))

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (elu(z + h) - elu(z - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_elu_backward_custom_alpha():
    def elu(z, alpha=0.5):
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))

    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z, usegrad=True)

    result_tensor = f.elu(z_tensor, alpha=0.5)
    result_tensor.backward(nura.tensor(np.ones_like(z)))

    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (elu(z + h, alpha=0.5) - elu(z - h, alpha=0.5)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_gelu_backward_scalar():
    def gelu(z):
        return (
            0.5
            * z
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))
        )

    z = np.random.randn()
    z_tensor = nura.tensor(z, usegrad=True)
    result_tensor = f.gelu(z_tensor)
    result_tensor.backward()
    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (gelu(z + h) - gelu(z - h)) / (2 * h)
    np.testing.assert_almost_equal(grad.data, expected_grad, decimal=5)


def test_gelu_backward_vector():
    def gelu(z):
        return (
            0.5
            * z
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))
        )

    z = np.random.randn(5)
    z_tensor = nura.tensor(z, usegrad=True)
    result_tensor = f.gelu(z_tensor)
    result_tensor.backward(nura.tensor(np.ones_like(z)))
    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (gelu(z + h) - gelu(z - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)


def test_gelu_backward_matrix():
    def gelu(z):
        return (
            0.5
            * z
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))
        )

    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z, usegrad=True)
    result_tensor = f.gelu(z_tensor)
    result_tensor.backward(nura.tensor(np.ones_like(z)))
    grad = z_tensor.grad
    h = 1e-8
    expected_grad = (gelu(z + h) - gelu(z - h)) / (2 * h)
    np.testing.assert_array_almost_equal(grad.data, expected_grad, decimal=5)
