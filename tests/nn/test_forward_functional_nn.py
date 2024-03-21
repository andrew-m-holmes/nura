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


def test_softmax_forward_scalar():
    z = np.random.randn()

    z_tensor = nura.tensor(z)
    result_tensor = f.softmax(z_tensor)
    result = result_tensor.data
    expected = np.exp(z) / np.sum(np.exp(z))
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_softmax_forward_vector():
    z = np.random.randn(5)

    z_tensor = nura.tensor(z)
    result_tensor = f.softmax(z_tensor)
    result = result_tensor.data
    expected = np.exp(z) / np.sum(np.exp(z))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_softmax_forward_matrix():
    z = np.random.randn(3, 3)

    z_tensor = nura.tensor(z)
    result_tensor = f.softmax(z_tensor, dim=-1)
    result = result_tensor.data
    expected = np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_softmax_forward_tensor_rank3():
    z = np.random.randn(3, 3, 3)

    z_tensor = nura.tensor(z)
    result_tensor = f.softmax(z_tensor, dim=1)
    result = result_tensor.data
    expected = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_softmax_forward_tensor_rank4():
    z = np.random.randn(4, 4, 4, 4)

    z_tensor = nura.tensor(z)
    result_tensor = f.softmax(z_tensor, dim=2)
    result = result_tensor.data
    expected = np.exp(z) / np.sum(np.exp(z), axis=2, keepdims=True)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_linear_forward_vector():
    x = np.random.randn(5)
    w = np.random.randn(5, 5)
    b = np.random.randn(5)

    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    b_tensor = nura.tensor(b)
    result_tensor = f.linear(x_tensor, w_tensor, b_tensor)
    result = result_tensor.data
    expected = x @ w.T + b
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_linear_forward_matrix():
    x = np.random.randn(3, 3)
    w = np.random.randn(3, 3)
    b = np.random.randn(3)

    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    b_tensor = nura.tensor(b)
    result_tensor = f.linear(x_tensor, w_tensor, b_tensor)
    result = result_tensor.data
    expected = x @ w.T + b
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_linear_forward_matrix_batch():
    x = np.random.randn(10, 3, 3)
    w = np.random.randn(3, 3)
    b = np.random.randn(3)

    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    b_tensor = nura.tensor(b)
    result_tensor = f.linear(x_tensor, w_tensor, b_tensor)
    result = result_tensor.data
    expected = x @ w.T + b
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_linear_forward_matrix_batch2():
    x = np.random.randn(10, 5, 3, 3)
    w = np.random.randn(3, 3)
    b = np.random.randn(3)

    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    b_tensor = nura.tensor(b)
    result_tensor = f.linear(x_tensor, w_tensor, b_tensor)
    result = result_tensor.data
    expected = x @ w.T + b
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_linear_forward_vector_no_bias():
    x = np.random.randn(5)
    w = np.random.randn(5, 5)

    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    result_tensor = f.linear(x_tensor, w_tensor)
    result = result_tensor.data
    expected = x @ w.T
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_linear_forward_matrix_no_bias():
    x = np.random.randn(3, 3)
    w = np.random.randn(3, 3)

    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    result_tensor = f.linear(x_tensor, w_tensor)
    result = result_tensor.data
    expected = x @ w.T
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_linear_forward_matrix_batch_no_bias():
    x = np.random.randn(10, 3, 3)
    w = np.random.randn(3, 3)

    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    result_tensor = f.linear(x_tensor, w_tensor)
    result = result_tensor.data
    expected = x @ w.T
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_linear_forward_matrix_batch2_no_bias():
    x = np.random.randn(10, 5, 3, 3)
    w = np.random.randn(3, 3)

    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    result_tensor = f.linear(x_tensor, w_tensor)
    result = result_tensor.data
    expected = x @ w.T
    np.testing.assert_array_almost_equal(result, expected, decimal=5)
