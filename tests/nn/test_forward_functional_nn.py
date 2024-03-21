import numpy as np
import nura
import nura.nn.functional as f


def test_relu_forward_scalar():
    z = np.random.randn()

    z_tensor = nura.tensor(z)
    result_tensor = f.relu(z_tensor)
    result = result_tensor.data
    expected = np.maximum(0, z)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_relu_forward_vector():
    z = np.random.randn(5)

    z_tensor = nura.tensor(z)
    result_tensor = f.relu(z_tensor)
    result = result_tensor.data
    expected = np.maximum(0, z)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_relu_forward_matrix():
    z = np.random.randn(3, 3)

    z_tensor = nura.tensor(z)
    result_tensor = f.relu(z_tensor)
    result = result_tensor.data
    expected = np.maximum(0, z)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_relu6_forward_scalar():
    z = np.random.randn()

    z_tensor = nura.tensor(z)
    result_tensor = f.relu6(z_tensor)
    result = result_tensor.data
    expected = np.minimum(np.maximum(0, z), 6)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_relu6_forward_vector():
    z = np.random.randn(5)

    z_tensor = nura.tensor(z)
    result_tensor = f.relu6(z_tensor)
    result = result_tensor.data
    expected = np.minimum(np.maximum(0, z), 6)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_relu6_forward_matrix():
    z = np.random.randn(3, 3)

    z_tensor = nura.tensor(z)
    result_tensor = f.relu6(z_tensor)
    result = result_tensor.data
    expected = np.minimum(np.maximum(0, z), 6)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_leakyrelu_forward_scalar():
    z = np.random.randn()

    z_tensor = nura.tensor(z)
    result_tensor = f.leakyrelu(z_tensor)
    result = result_tensor.data
    expected = np.maximum(0.01 * z, z)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_leakyrelu_forward_vector():
    z = np.random.randn(5)

    z_tensor = nura.tensor(z)
    result_tensor = f.leakyrelu(z_tensor)
    result = result_tensor.data
    expected = np.maximum(0.01 * z, z)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_leakyrelu_forward_matrix():
    z = np.random.randn(3, 3)

    z_tensor = nura.tensor(z)
    result_tensor = f.leakyrelu(z_tensor)
    result = result_tensor.data
    expected = np.maximum(0.01 * z, z)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_elu_forward_scalar():
    z = np.random.randn()

    z_tensor = nura.tensor(z)
    result_tensor = f.elu(z_tensor)
    result = result_tensor.data
    expected = np.where(z > 0, z, 1.0 * (np.exp(z) - 1))
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_elu_forward_vector():
    z = np.random.randn(5)

    z_tensor = nura.tensor(z)
    result_tensor = f.elu(z_tensor)
    result = result_tensor.data
    expected = np.where(z > 0, z, 1.0 * (np.exp(z) - 1))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_elu_forward_matrix():
    z = np.random.randn(3, 3)

    z_tensor = nura.tensor(z)
    result_tensor = f.elu(z_tensor)
    result = result_tensor.data
    expected = np.where(z > 0, z, 1.0 * (np.exp(z) - 1))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_gelu_forward_scalar():
    z = np.random.randn()  # Random scalar
    z_tensor = nura.tensor(z)
    result_tensor = f.gelu(z_tensor)
    result = result_tensor.data
    expected = (
        0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))
    )
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_gelu_forward_vector():
    z = np.random.randn(5)  # Random vector
    z_tensor = nura.tensor(z)
    result_tensor = f.gelu(z_tensor)
    result = result_tensor.data
    expected = (
        0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))
    )
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_gelu_forward_matrix():
    z = np.random.randn(3, 3)  # Random matrix
    z_tensor = nura.tensor(z)
    result_tensor = f.gelu(z_tensor)
    result = result_tensor.data
    expected = (
        0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))
    )
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_sigmoid_forward_scalar():
    z = np.random.randn()

    z_tensor = nura.tensor(z)
    result_tensor = f.sigmoid(z_tensor)
    result = result_tensor.data
    expected = 1 / (1 + np.exp(-z))
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_sigmoid_forward_vector():
    z = np.random.randn(5)

    z_tensor = nura.tensor(z)
    result_tensor = f.sigmoid(z_tensor)
    result = result_tensor.data
    expected = 1 / (1 + np.exp(-z))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_sigmoid_forward_matrix():
    z = np.random.randn(3, 3)

    z_tensor = nura.tensor(z)
    result_tensor = f.sigmoid(z_tensor)
    result = result_tensor.data
    expected = 1 / (1 + np.exp(-z))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_tanh_forward_scalar():
    z = np.random.randn()

    z_tensor = nura.tensor(z)
    result_tensor = f.tanh(z_tensor)
    result = result_tensor.data
    expected = np.tanh(z)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_tanh_forward_vector():
    z = np.random.randn(5)

    z_tensor = nura.tensor(z)
    result_tensor = f.tanh(z_tensor)
    result = result_tensor.data
    expected = np.tanh(z)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_tanh_forward_matrix():
    z = np.random.randn(3, 3)

    z_tensor = nura.tensor(z)
    result_tensor = f.tanh(z_tensor)
    result = result_tensor.data
    expected = np.tanh(z)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)
