import numpy as np
import nura
import nura.nn.functional as f


def test_linear_vector_no_bias():
    x = np.random.rand(4)
    w = np.random.rand(3, 4)
    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    result_tensor = f.linear(x_tensor, w_tensor)

    expected_result = np.dot(x, w.T)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_linear_vector_with_bias():
    x = np.random.rand(4)
    w = np.random.rand(3, 4)
    b = np.random.rand(3)
    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    b_tensor = nura.tensor(b)
    result_tensor = f.linear(x_tensor, w_tensor, b_tensor)

    expected_result = np.dot(x, w.T) + b

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_linear_matrix_no_bias():
    x = np.random.rand(2, 4)
    w = np.random.rand(3, 4)
    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    result_tensor = f.linear(x_tensor, w_tensor)

    expected_result = np.dot(x, w.T)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_linear_matrix_with_bias():
    x = np.random.rand(2, 4)
    w = np.random.rand(3, 4)
    b = np.random.rand(3)
    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    b_tensor = nura.tensor(b)
    result_tensor = f.linear(x_tensor, w_tensor, b_tensor)

    expected_result = np.dot(x, w.T) + b

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_linear_tensor_no_bias():
    x = np.random.rand(2, 3, 4)
    w = np.random.rand(5, 4)
    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    result_tensor = f.linear(x_tensor, w_tensor)

    expected_result = np.matmul(x, w.T)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_linear_tensor_with_bias():
    x = np.random.rand(2, 3, 4)
    w = np.random.rand(5, 4)
    b = np.random.rand(5)
    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    b_tensor = nura.tensor(b)
    result_tensor = f.linear(x_tensor, w_tensor, b_tensor)

    expected_result = np.matmul(x, w.T) + b

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_linear_higher_order_tensor_no_bias():
    x = np.random.rand(2, 3, 4, 5)
    w = np.random.rand(7, 5)
    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    result_tensor = f.linear(x_tensor, w_tensor)

    expected_result = np.matmul(x, w.T)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_linear_higher_order_tensor_with_bias():
    x = np.random.rand(2, 3, 4, 5)
    w = np.random.rand(7, 5)
    b = np.random.rand(7)
    x_tensor = nura.tensor(x)
    w_tensor = nura.tensor(w)
    b_tensor = nura.tensor(b)
    result_tensor = f.linear(x_tensor, w_tensor, b_tensor)

    expected_result = np.matmul(x, w.T) + b

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


# TODO (linear test for 3D or higher weight)


def test_sigmoid_scalar():
    x = np.array(0.5)
    x_tensor = nura.tensor(x)
    result_tensor = f.sigmoid(x_tensor)

    expected_result = 1 / (1 + np.exp(-x))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_sigmoid_vector():
    x = np.random.rand(5)
    x_tensor = nura.tensor(x)
    result_tensor = f.sigmoid(x_tensor)

    expected_result = 1 / (1 + np.exp(-x))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_sigmoid_matrix():
    x = np.random.rand(3, 4)
    x_tensor = nura.tensor(x)
    result_tensor = f.sigmoid(x_tensor)

    expected_result = 1 / (1 + np.exp(-x))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_sigmoid_tensor():
    x = np.random.rand(2, 3, 4)
    x_tensor = nura.tensor(x)
    result_tensor = f.sigmoid(x_tensor)

    expected_result = 1 / (1 + np.exp(-x))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_sigmoid_higher_order_tensor():
    x = np.random.rand(2, 3, 4, 5)
    x_tensor = nura.tensor(x)
    result_tensor = f.sigmoid(x_tensor)

    expected_result = 1 / (1 + np.exp(-x))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_tanh_scalar():
    x = np.array(-0.5)
    x_tensor = nura.tensor(x)
    result_tensor = f.tanh(x_tensor)

    expected_result = np.tanh(x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_tanh_vector():
    x = np.random.rand(4)
    x_tensor = nura.tensor(x)
    result_tensor = f.tanh(x_tensor)

    expected_result = np.tanh(x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_tanh_matrix():
    x = np.random.rand(2, 3)
    x_tensor = nura.tensor(x)
    result_tensor = f.tanh(x_tensor)

    expected_result = np.tanh(x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_tanh_tensor():
    x = np.random.rand(3, 3, 2)
    x_tensor = nura.tensor(x)
    result_tensor = f.tanh(x_tensor)

    expected_result = np.tanh(x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_tanh_higher_order_tensor():
    x = np.random.rand(2, 4, 2, 3)
    x_tensor = nura.tensor(x)
    result_tensor = f.tanh(x_tensor)

    expected_result = np.tanh(x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_relu_scalar():
    x = np.array(-0.2)
    x_tensor = nura.tensor(x)
    result_tensor = f.relu(x_tensor)

    expected_result = np.maximum(0, x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_relu_vector():
    x = np.random.randn(5)
    x_tensor = nura.tensor(x)
    result_tensor = f.relu(x_tensor)

    expected_result = np.maximum(0, x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_relu_matrix():
    x = np.random.randn(3, 2)
    x_tensor = nura.tensor(x)
    result_tensor = f.relu(x_tensor)

    expected_result = np.maximum(0, x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_relu_tensor():
    x = np.random.randn(4, 3, 2)
    x_tensor = nura.tensor(x)
    result_tensor = f.relu(x_tensor)

    expected_result = np.maximum(0, x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_relu_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 2)
    x_tensor = nura.tensor(x)
    result_tensor = f.relu(x_tensor)

    expected_result = np.maximum(0, x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_relu6_scalar():
    x = np.array(6.5)
    x_tensor = nura.tensor(x)
    result_tensor = f.relu6(x_tensor)

    expected_result = np.minimum(np.maximum(0, x), 6)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_relu6_vector():
    x = np.random.randn(6) * 2
    x_tensor = nura.tensor(x)
    result_tensor = f.relu6(x_tensor)

    expected_result = np.minimum(np.maximum(0, x), 6)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_relu6_matrix():
    x = np.random.randn(3, 5) * 3
    x_tensor = nura.tensor(x)
    result_tensor = f.relu6(x_tensor)

    expected_result = np.minimum(np.maximum(0, x), 6)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_relu6_tensor():
    x = np.random.randn(4, 3, 2) * 4
    x_tensor = nura.tensor(x)
    result_tensor = f.relu6(x_tensor)

    expected_result = np.minimum(np.maximum(0, x), 6)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_relu6_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 2) * 5
    x_tensor = nura.tensor(x)
    result_tensor = f.relu6(x_tensor)

    expected_result = np.minimum(np.maximum(0, x), 6)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_leakyrelu_scalar():
    x = np.array(-1.0)
    x_tensor = nura.tensor(x)
    result_tensor = f.leakyrelu(x_tensor)

    expected_result = np.where(x > 0, x, x * 0.01)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_leakyrelu_vector():
    x = np.random.randn(5)
    x_tensor = nura.tensor(x)
    result_tensor = f.leakyrelu(x_tensor)

    expected_result = np.where(x > 0, x, x * 0.01)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_leakyrelu_matrix():
    x = np.random.randn(3, 4)
    x_tensor = nura.tensor(x)
    result_tensor = f.leakyrelu(x_tensor)

    expected_result = np.where(x > 0, x, x * 0.01)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_leakyrelu_tensor():
    x = np.random.randn(2, 3, 4)
    x_tensor = nura.tensor(x)
    result_tensor = f.leakyrelu(x_tensor)

    expected_result = np.where(x > 0, x, x * 0.01)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_leakyrelu_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 5)
    x_tensor = nura.tensor(x)
    result_tensor = f.leakyrelu(x_tensor)

    expected_result = np.where(x > 0, x, x * 0.01)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_leakyrelu_vector_custom_alpha():
    x = np.random.randn(5)
    alpha = 0.1
    x_tensor = nura.tensor(x)
    result_tensor = f.leakyrelu(x_tensor, alpha)

    expected_result = np.where(x > 0, x, x * alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_leakyrelu_matrix_custom_alpha():
    x = np.random.randn(3, 4)
    alpha = 0.2
    x_tensor = nura.tensor(x)
    result_tensor = f.leakyrelu(x_tensor, alpha)

    expected_result = np.where(x > 0, x, x * alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_leakyrelu_tensor_custom_alpha():
    x = np.random.randn(2, 3, 4)
    alpha = 0.05
    x_tensor = nura.tensor(x)
    result_tensor = f.leakyrelu(x_tensor, alpha)

    expected_result = np.where(x > 0, x, x * alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_elu_scalar():
    x = np.array(-1.0)
    x_tensor = nura.tensor(x)
    result_tensor = f.elu(x_tensor)

    expected_result = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_elu_vector():
    x = np.random.randn(5)
    x_tensor = nura.tensor(x)
    result_tensor = f.elu(x_tensor)

    expected_result = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_elu_matrix():
    x = np.random.randn(3, 4)
    x_tensor = nura.tensor(x)
    result_tensor = f.elu(x_tensor)

    expected_result = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_elu_tensor():
    x = np.random.randn(2, 3, 4)
    x_tensor = nura.tensor(x)
    result_tensor = f.elu(x_tensor)

    expected_result = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_elu_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 5)
    x_tensor = nura.tensor(x)
    result_tensor = f.elu(x_tensor)

    expected_result = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_elu_vector_custom_alpha():
    x = np.random.randn(5)
    alpha = 0.5
    x_tensor = nura.tensor(x)
    result_tensor = f.elu(x_tensor, alpha)

    expected_result = np.where(x > 0, x, alpha * (np.exp(x) - 1))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_elu_matrix_custom_alpha():
    x = np.random.randn(3, 4)
    alpha = 1.5
    x_tensor = nura.tensor(x)
    result_tensor = f.elu(x_tensor, alpha)

    expected_result = np.where(x > 0, x, alpha * (np.exp(x) - 1))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_elu_tensor_custom_alpha():
    x = np.random.randn(2, 3, 4)
    alpha = 2.0
    x_tensor = nura.tensor(x)
    result_tensor = f.elu(x_tensor, alpha)

    expected_result = np.where(x > 0, x, alpha * (np.exp(x) - 1))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def gelu_reference(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def test_gelu_scalar():
    x = np.array(0.5)
    x_tensor = nura.tensor(x)
    result_tensor = f.gelu(x_tensor)

    expected_result = gelu_reference(x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_gelu_vector():
    x = np.random.randn(4)
    x_tensor = nura.tensor(x)
    result_tensor = f.gelu(x_tensor)

    expected_result = gelu_reference(x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_gelu_matrix():
    x = np.random.randn(3, 3)
    x_tensor = nura.tensor(x)
    result_tensor = f.gelu(x_tensor)

    expected_result = gelu_reference(x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_gelu_tensor():
    x = np.random.randn(2, 4, 3)
    x_tensor = nura.tensor(x)
    result_tensor = f.gelu(x_tensor)

    expected_result = gelu_reference(x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_gelu_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 2)
    x_tensor = nura.tensor(x)
    result_tensor = f.gelu(x_tensor)

    expected_result = gelu_reference(x)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def celu_reference(x, alpha):
    return np.where(x > 0, x, alpha * (np.exp(x / alpha) - 1))


def test_celu_scalar():
    x = np.array(-0.5)
    alpha = 1.0
    x_tensor = nura.tensor(x)
    result_tensor = f.celu(x_tensor, alpha)

    expected_result = celu_reference(x, alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_celu_vector():
    x = np.random.randn(5)
    alpha = 1.0
    x_tensor = nura.tensor(x)
    result_tensor = f.celu(x_tensor, alpha)

    expected_result = celu_reference(x, alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_celu_matrix():
    x = np.random.randn(3, 3)
    alpha = 1.0
    x_tensor = nura.tensor(x)
    result_tensor = f.celu(x_tensor, alpha)

    expected_result = celu_reference(x, alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_celu_tensor():
    x = np.random.randn(2, 4, 3)
    alpha = 1.0
    x_tensor = nura.tensor(x)
    result_tensor = f.celu(x_tensor, alpha)

    expected_result = celu_reference(x, alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_celu_higher_order_tensor():
    x = np.random.randn(2, 3, 4, 2)
    alpha = 1.0
    x_tensor = nura.tensor(x)
    result_tensor = f.celu(x_tensor, alpha)

    expected_result = celu_reference(x, alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_celu_vector_custom_alpha():
    x = np.random.randn(5)
    alpha = 0.5
    x_tensor = nura.tensor(x)
    result_tensor = f.celu(x_tensor, alpha)

    expected_result = celu_reference(x, alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_celu_matrix_custom_alpha():
    x = np.random.randn(3, 4)
    alpha = 2.0
    x_tensor = nura.tensor(x)
    result_tensor = f.celu(x_tensor, alpha)

    expected_result = celu_reference(x, alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_celu_tensor_custom_alpha():
    x = np.random.randn(2, 3, 4)
    alpha = 1.5
    x_tensor = nura.tensor(x)
    result_tensor = f.celu(x_tensor, alpha)

    expected_result = celu_reference(x, alpha)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def softmax_reference(x, dim):
    exp_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return exp_x / np.sum(exp_x, axis=dim, keepdims=True)


def test_softmax_vector():
    x = np.random.rand(5)
    x_tensor = nura.tensor(x)
    result_tensor = f.softmax(x_tensor)

    expected_result = softmax_reference(x, dim=-1)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_softmax_matrix():
    x = np.random.rand(3, 4)
    x_tensor = nura.tensor(x)
    result_tensor = f.softmax(x_tensor)

    expected_result = softmax_reference(x, dim=-1)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_softmax_tensor():
    x = np.random.rand(2, 3, 4)
    x_tensor = nura.tensor(x)
    result_tensor = f.softmax(x_tensor)

    expected_result = softmax_reference(x, dim=-1)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_softmax_higher_order_tensor():
    x = np.random.rand(2, 3, 4, 5)
    x_tensor = nura.tensor(x)
    result_tensor = f.softmax(x_tensor)

    expected_result = softmax_reference(x, dim=-1)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_softmax_matrix_dim_0():
    x = np.random.rand(3, 4)
    x_tensor = nura.tensor(x)
    result_tensor = f.softmax(x_tensor, dim=0)

    expected_result = softmax_reference(x, dim=0)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_softmax_tensor_dim_1():
    x = np.random.rand(2, 3, 4)
    x_tensor = nura.tensor(x)
    result_tensor = f.softmax(x_tensor, dim=1)

    expected_result = softmax_reference(x, dim=1)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_softmax_higher_order_tensor_dim_2():
    x = np.random.rand(2, 3, 4, 5)
    x_tensor = nura.tensor(x)
    result_tensor = f.softmax(x_tensor, dim=2)

    expected_result = softmax_reference(x, dim=2)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_softmax_higher_order_tensor_dim_0():
    x = np.random.rand(2, 3, 4, 5)
    x_tensor = nura.tensor(x)
    result_tensor = f.softmax(x_tensor, dim=0)

    expected_result = softmax_reference(x, dim=0)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def attention_reference(q, k, v, mask=None, maskfill=-1e9):
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(k.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, maskfill)
    weights = softmax_reference(scores, dim=-1)
    context = np.matmul(weights, v)
    return context, weights


def test_attention_basic():
    q = np.random.rand(2, 3, 4)
    k = np.random.rand(2, 3, 4)
    v = np.random.rand(2, 3, 4)
    q_tensor = nura.tensor(q)
    k_tensor = nura.tensor(k)
    v_tensor = nura.tensor(v)
    context, weights = f.attention(q_tensor, k_tensor, v_tensor)

    expected_context, expected_weights = attention_reference(q, k, v)

    np.testing.assert_allclose(context.data, expected_context, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(weights.data, expected_weights, rtol=1e-7, atol=1e-7)


def test_attention_with_mask():
    q = np.random.rand(2, 3, 4)
    k = np.random.rand(2, 3, 4)
    v = np.random.rand(2, 3, 4)
    mask = np.random.choice([True, False], size=(2, 3, 3))
    q_tensor = nura.tensor(q)
    k_tensor = nura.tensor(k)
    v_tensor = nura.tensor(v)
    mask_tensor = nura.tensor(mask)
    context, weights = f.attention(q_tensor, k_tensor, v_tensor, mask_tensor)

    expected_context, expected_weights = attention_reference(q, k, v, mask)

    np.testing.assert_allclose(context.data, expected_context, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(weights.data, expected_weights, rtol=1e-7, atol=1e-7)


def test_attention_with_pad_mask():
    q = np.random.rand(2, 3, 4)
    k = np.random.rand(2, 3, 4)
    v = np.random.rand(2, 3, 4)
    pad_id = 0
    mask = np.random.choice([True, False], size=(2, 3, 3))
    mask[:, pad_id] = False
    q_tensor = nura.tensor(q)
    k_tensor = nura.tensor(k)
    v_tensor = nura.tensor(v)
    mask_tensor = nura.tensor(mask)
    context, weights = f.attention(q_tensor, k_tensor, v_tensor, mask_tensor)

    expected_context, expected_weights = attention_reference(q, k, v, mask)

    np.testing.assert_allclose(context.data, expected_context, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(weights.data, expected_weights, rtol=1e-7, atol=1e-7)


def test_attention_no_peak():
    q = np.random.rand(2, 3, 4)
    k = np.random.rand(2, 3, 4)
    v = np.random.rand(2, 3, 4)
    mask = np.triu(np.ones((3, 3)), k=1).astype(bool)
    q_tensor = nura.tensor(q)
    k_tensor = nura.tensor(k)
    v_tensor = nura.tensor(v)
    mask_tensor = nura.tensor(mask)
    context, weights = f.attention(q_tensor, k_tensor, v_tensor, mask_tensor)

    expected_context, expected_weights = attention_reference(q, k, v, mask)

    np.testing.assert_allclose(context.data, expected_context, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(weights.data, expected_weights, rtol=1e-7, atol=1e-7)


def test_attention_with_dropout():
    q = np.random.rand(2, 3, 4)
    k = np.random.rand(2, 3, 4)
    v = np.random.rand(2, 3, 4)
    mask = np.triu(np.ones((3, 3)), k=1).astype(bool)
    q_tensor = nura.tensor(q)
    k_tensor = nura.tensor(k)
    v_tensor = nura.tensor(v)
    mask_tensor = nura.tensor(mask)
    context, weights = f.attention(q_tensor, k_tensor, v_tensor, mask_tensor, drop=0.1)

    assert np.any(weights.data == 0)


def test_attention_basic_different_lengths():
    q = np.random.rand(2, 4, 4)
    k = np.random.rand(2, 3, 4)
    v = np.random.rand(2, 3, 4)
    q_tensor = nura.tensor(q)
    k_tensor = nura.tensor(k)
    v_tensor = nura.tensor(v)
    context, weights = f.attention(q_tensor, k_tensor, v_tensor)

    expected_context, expected_weights = attention_reference(q, k, v)

    np.testing.assert_allclose(context.data, expected_context, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(weights.data, expected_weights, rtol=1e-7, atol=1e-7)


def test_attention_with_mask_different_lengths():
    q = np.random.rand(2, 4, 4)
    k = np.random.rand(2, 3, 4)
    v = np.random.rand(2, 3, 4)
    mask = np.random.choice([True, False], size=(2, 4, 3))
    q_tensor = nura.tensor(q)
    k_tensor = nura.tensor(k)
    v_tensor = nura.tensor(v)
    mask_tensor = nura.tensor(mask)
    context, weights = f.attention(q_tensor, k_tensor, v_tensor, mask_tensor)

    expected_context, expected_weights = attention_reference(q, k, v, mask)

    np.testing.assert_allclose(context.data, expected_context, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(weights.data, expected_weights, rtol=1e-7, atol=1e-7)


def test_attention_with_pad_mask_different_lengths():
    q = np.random.rand(2, 4, 4)
    k = np.random.rand(2, 3, 4)
    v = np.random.rand(2, 3, 4)
    pad_id = 0
    mask = np.random.choice([True, False], size=(2, 4, 3))
    mask[:, pad_id, :] = False
    q_tensor = nura.tensor(q)
    k_tensor = nura.tensor(k)
    v_tensor = nura.tensor(v)
    mask_tensor = nura.tensor(mask)
    context, weights = f.attention(q_tensor, k_tensor, v_tensor, mask_tensor)

    expected_context, expected_weights = attention_reference(q, k, v, mask)

    np.testing.assert_allclose(context.data, expected_context, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(weights.data, expected_weights, rtol=1e-7, atol=1e-7)


def test_attention_no_peak_different_lengths():
    q = np.random.rand(2, 4, 4)
    k = np.random.rand(2, 3, 4)
    v = np.random.rand(2, 3, 4)
    mask = np.triu(np.ones((4, 3)), k=1).astype(bool)
    q_tensor = nura.tensor(q)
    k_tensor = nura.tensor(k)
    v_tensor = nura.tensor(v)
    mask_tensor = nura.tensor(mask)
    context, weights = f.attention(q_tensor, k_tensor, v_tensor, mask_tensor)

    expected_context, expected_weights = attention_reference(q, k, v, mask)

    np.testing.assert_allclose(context.data, expected_context, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(weights.data, expected_weights, rtol=1e-7, atol=1e-7)


def test_attention_with_dropout_different_lengths():
    q = np.random.rand(2, 4, 4)
    k = np.random.rand(2, 3, 4)
    v = np.random.rand(2, 3, 4)
    mask = np.triu(np.ones((4, 3)), k=1).astype(bool)
    q_tensor = nura.tensor(q)
    k_tensor = nura.tensor(k)
    v_tensor = nura.tensor(v)
    mask_tensor = nura.tensor(mask)
    context, weights = f.attention(q_tensor, k_tensor, v_tensor, mask_tensor, drop=0.1)

    assert np.any(weights.data == 0)


def test_embedding_vector():
    x = np.array([1, 2, 3])
    w = np.random.rand(5, 4)
    x_tensor = nura.tensor(x, dtype=nura.int)
    w_tensor = nura.tensor(w)
    result_tensor = f.embedding(x_tensor, w_tensor)

    expected_result = w[x]

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_embedding_batch_vectors():
    x = np.array([[1, 2, 3], [0, 2, 4]])
    w = np.random.rand(5, 4)
    x_tensor = nura.tensor(x, dtype=nura.int)
    w_tensor = nura.tensor(w)
    result_tensor = f.embedding(x_tensor, w_tensor)

    expected_result = w[x]

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_embedding_vector_with_padid():
    x = np.array([1, 2, 3, 0])
    w = np.random.rand(5, 4)
    x_tensor = nura.tensor(x, dtype=nura.int)
    w_tensor = nura.tensor(w)
    result_tensor = f.embedding(x_tensor, w_tensor, padid=0)

    expected_result = w[x]
    expected_result[x == 0] = 0

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_embedding_batch_vectors_with_padid():
    x = np.array([[1, 2, 3], [0, 2, 4]])
    w = np.random.rand(5, 4)
    x_tensor = nura.tensor(x, dtype=nura.int)
    w_tensor = nura.tensor(w)
    result_tensor = f.embedding(x_tensor, w_tensor, padid=0)

    expected_result = w[x]
    expected_result[x == 0] = 0

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def binarycrossentropy_reference(x, y, reduction):
    loss = -(y * np.log(x) + (1 - y) * np.log(1 - x))
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    return loss


def test_binarycrossentropy_vector_mean():
    x = np.array([0.9, 0.8, 0.1])
    y = np.array([1.0, 0.0, 1.0])
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.binarycrossentropy(x_tensor, y_tensor, reduction="mean")

    expected_result = binarycrossentropy_reference(x, y, reduction="mean")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_binarycrossentropy_vector_sum():
    x = np.array([0.9, 0.8, 0.1])
    y = np.array([1.0, 0.0, 1.0])
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.binarycrossentropy(x_tensor, y_tensor, reduction="sum")

    expected_result = binarycrossentropy_reference(x, y, reduction="sum")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_binarycrossentropy_vector_none():
    x = np.array([0.9, 0.8, 0.1])
    y = np.array([1.0, 0.0, 1.0])
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.binarycrossentropy(x_tensor, y_tensor, reduction=None)

    expected_result = binarycrossentropy_reference(x, y, reduction=None)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_binarycrossentropy_batch_vectors_mean():
    x = np.array([[0.9, 0.8, 0.1], [0.7, 0.4, 0.3]])
    y = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.binarycrossentropy(x_tensor, y_tensor, reduction="mean")

    expected_result = binarycrossentropy_reference(x, y, reduction="mean")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_binarycrossentropy_batch_vectors_sum():
    x = np.array([[0.9, 0.8, 0.1], [0.7, 0.4, 0.3]])
    y = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.binarycrossentropy(x_tensor, y_tensor, reduction="sum")

    expected_result = binarycrossentropy_reference(x, y, reduction="sum")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_binarycrossentropy_batch_vectors_none():
    x = np.array([[0.9, 0.8, 0.1], [0.7, 0.4, 0.3]])
    y = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.binarycrossentropy(x_tensor, y_tensor, reduction=None)

    expected_result = binarycrossentropy_reference(x, y, reduction=None)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def crossentropy_reference(x, y, ignoreid=None, reduction=None):
    x = x - np.max(x, axis=1, keepdims=True)
    log_softmax_x = x - np.log(np.sum(np.exp(x), axis=1, keepdims=True))
    nll_loss = -log_softmax_x[np.arange(len(y)), y]
    if ignoreid is not None:
        mask = y != ignoreid
        nll_loss = nll_loss[mask]
    if reduction == "mean":
        return np.mean(nll_loss)
    elif reduction == "sum":
        return np.sum(nll_loss)
    return nll_loss


def test_crossentropy_basic_mean():
    x = np.random.rand(3, 5)
    y = np.array([0, 2, 1])
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y, dtype=nura.int)
    result_tensor = f.crossentropy(x_tensor, y_tensor, reduction="mean")

    expected_result = crossentropy_reference(x, y, reduction="mean")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_crossentropy_basic_sum():
    x = np.random.rand(3, 5)
    y = np.array([0, 2, 1])
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y, dtype=nura.int)
    result_tensor = f.crossentropy(x_tensor, y_tensor, reduction="sum")

    expected_result = crossentropy_reference(x, y, reduction="sum")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_crossentropy_basic_none():
    x = np.random.rand(3, 5)
    y = np.array([0, 2, 1])
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y, dtype=nura.int)
    result_tensor = f.crossentropy(x_tensor, y_tensor, reduction=None)

    expected_result = crossentropy_reference(x, y, reduction=None)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_crossentropy_with_ignoreid_mean():
    x = np.random.rand(4, 5)
    y = np.array([0, 2, 1, 3])
    ignoreid = 2
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y, dtype=nura.int)
    result_tensor = f.crossentropy(
        x_tensor, y_tensor, ignoreid=ignoreid, reduction="mean"
    )

    expected_result = crossentropy_reference(x, y, ignoreid=ignoreid, reduction="mean")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_crossentropy_with_ignoreid_sum():
    x = np.random.rand(4, 5)
    y = np.array([0, 2, 1, 3])
    ignoreid = 2
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y, dtype=nura.int)
    result_tensor = f.crossentropy(
        x_tensor, y_tensor, ignoreid=ignoreid, reduction="sum"
    )

    expected_result = crossentropy_reference(x, y, ignoreid=ignoreid, reduction="sum")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_crossentropy_with_ignoreid_none():
    x = np.random.rand(4, 5)
    y = np.array([0, 2, 1, 3])
    ignoreid = 2
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y, dtype=nura.int)
    result_tensor = f.crossentropy(
        x_tensor, y_tensor, ignoreid=ignoreid, reduction=None
    )

    expected_result = crossentropy_reference(x, y, ignoreid=ignoreid, reduction=None)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def mse_reference(x, y, reduction=None):
    loss = 0.5 * np.power((x - y), 2)
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    return loss


def test_mse_scalar_mean():
    x = np.array(2.0)
    y = np.array(3.0)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction="mean")

    expected_result = mse_reference(x, y, reduction="mean")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_scalar_sum():
    x = np.array(2.0)
    y = np.array(3.0)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction="sum")

    expected_result = mse_reference(x, y, reduction="sum")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_scalar_none():
    x = np.array(2.0)
    y = np.array(3.0)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction=None)

    expected_result = mse_reference(x, y, reduction=None)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_vector_mean():
    x = np.random.rand(5)
    y = np.random.rand(5)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction="mean")

    expected_result = mse_reference(x, y, reduction="mean")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_vector_sum():
    x = np.random.rand(5)
    y = np.random.rand(5)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction="sum")

    expected_result = mse_reference(x, y, reduction="sum")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_vector_none():
    x = np.random.rand(5)
    y = np.random.rand(5)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction=None)

    expected_result = mse_reference(x, y, reduction=None)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_matrix_mean():
    x = np.random.rand(3, 4)
    y = np.random.rand(3, 4)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction="mean")

    expected_result = mse_reference(x, y, reduction="mean")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_matrix_sum():
    x = np.random.rand(3, 4)
    y = np.random.rand(3, 4)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction="sum")

    expected_result = mse_reference(x, y, reduction="sum")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_matrix_none():
    x = np.random.rand(3, 4)
    y = np.random.rand(3, 4)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction=None)

    expected_result = mse_reference(x, y, reduction=None)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_tensor_mean():
    x = np.random.rand(2, 3, 4)
    y = np.random.rand(2, 3, 4)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction="mean")

    expected_result = mse_reference(x, y, reduction="mean")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_tensor_sum():
    x = np.random.rand(2, 3, 4)
    y = np.random.rand(2, 3, 4)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction="sum")

    expected_result = mse_reference(x, y, reduction="sum")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_tensor_none():
    x = np.random.rand(2, 3, 4)
    y = np.random.rand(2, 3, 4)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction=None)

    expected_result = mse_reference(x, y, reduction=None)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_higher_order_tensor_mean():
    x = np.random.rand(2, 3, 4, 5)
    y = np.random.rand(2, 3, 4, 5)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction="mean")

    expected_result = mse_reference(x, y, reduction="mean")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_higher_order_tensor_sum():
    x = np.random.rand(2, 3, 4, 5)
    y = np.random.rand(2, 3, 4, 5)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction="sum")

    expected_result = mse_reference(x, y, reduction="sum")

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_mse_higher_order_tensor_none():
    x = np.random.rand(2, 3, 4, 5)
    y = np.random.rand(2, 3, 4, 5)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result_tensor = f.mse(x_tensor, y_tensor, reduction=None)

    expected_result = mse_reference(x, y, reduction=None)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_dropout_scalar():
    x = np.array(1.0)
    p = 0.5
    x_tensor = nura.tensor(x)
    result_tensor = f.dropout(x_tensor, p=p)

    assert result_tensor.data in [0.0, x / (1 - p)]


def test_dropout_vector():
    x = np.random.rand(5)
    p = 0.5
    x_tensor = nura.tensor(x)
    result_tensor = f.dropout(x_tensor, p=p)

    expected_values = x / (1 - p)
    assert np.all(
        (result_tensor.data == 0.0) | np.isclose(result_tensor.data, expected_values)
    )


def test_dropout_matrix():
    x = np.random.rand(3, 4)
    p = 0.5
    x_tensor = nura.tensor(x)
    result_tensor = f.dropout(x_tensor, p=p)

    expected_values = x / (1 - p)
    assert np.all(
        (result_tensor.data == 0.0) | np.isclose(result_tensor.data, expected_values)
    )


def test_dropout_tensor():
    x = np.random.rand(2, 3, 4)
    p = 0.5
    x_tensor = nura.tensor(x)
    result_tensor = f.dropout(x_tensor, p=p)

    expected_values = x / (1 - p)
    assert np.all(
        (result_tensor.data == 0.0) | np.isclose(result_tensor.data, expected_values)
    )


def test_dropout_higher_order_tensor():
    x = np.random.rand(2, 3, 4, 5)
    p = 0.5
    x_tensor = nura.tensor(x)
    result_tensor = f.dropout(x_tensor, p=p)

    expected_values = x / (1 - p)
    assert np.all(
        (result_tensor.data == 0.0) | np.isclose(result_tensor.data, expected_values)
    )


def test_dropout_high_rate():
    x = np.random.rand(2, 3, 4, 5)
    p = 0.9
    x_tensor = nura.tensor(x)
    result_tensor = f.dropout(x_tensor, p=p)

    assert np.any(result_tensor.data == 0.0)
    expected_values = x / (1 - p)
    assert np.all(
        (result_tensor.data == 0.0) | np.isclose(result_tensor.data, expected_values)
    )


def layernorm_reference(x, gamma, beta, dim, eps=1e-5):
    mean = np.mean(x, axis=dim, keepdims=True)
    var = np.var(x, axis=dim, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    return gamma * normalized + beta


def test_layernorm_vector():
    x = np.random.rand(5)
    gamma = np.ones(5)
    beta = np.zeros(5)
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result_tensor = f.layernorm(x_tensor, gamma_tensor, beta_tensor, dim=-1)

    expected_result = layernorm_reference(x, gamma, beta, dim=-1)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_layernorm_matrix():
    x = np.random.rand(3, 4)
    gamma = np.ones(4)
    beta = np.zeros(4)
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result_tensor = f.layernorm(x_tensor, gamma_tensor, beta_tensor, dim=-1)

    expected_result = layernorm_reference(x, gamma, beta, dim=-1)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_layernorm_tensor():
    x = np.random.rand(2, 3, 4)
    gamma = np.ones(4)
    beta = np.zeros(4)
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result_tensor = f.layernorm(x_tensor, gamma_tensor, beta_tensor, dim=-1)

    expected_result = layernorm_reference(x, gamma, beta, dim=-1)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_layernorm_higher_order_tensor():
    x = np.random.rand(2, 3, 4, 5)
    gamma = np.ones(5)
    beta = np.zeros(5)
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result_tensor = f.layernorm(x_tensor, gamma_tensor, beta_tensor, dim=-1)

    expected_result = layernorm_reference(x, gamma, beta, dim=-1)

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


def test_layernorm_tensor_with_tuple_dim():
    x = np.random.rand(2, 3, 4, 5)
    gamma = np.ones((4, 5))
    beta = np.zeros((4, 5))
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result_tensor = f.layernorm(x_tensor, gamma_tensor, beta_tensor, dim=(-2, -1))

    expected_result = layernorm_reference(x, gamma, beta, dim=(-2, -1))

    np.testing.assert_allclose(
        result_tensor.data, expected_result, rtol=1e-7, atol=1e-7
    )


# TODO add tests for batchnorm
