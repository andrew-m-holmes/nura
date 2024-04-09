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
    z = np.random.randn()
    z_tensor = nura.tensor(z)
    result_tensor = f.gelu(z_tensor)
    result = result_tensor.data
    expected = (
        0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))
    )
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_gelu_forward_vector():
    z = np.random.randn(5)
    z_tensor = nura.tensor(z)
    result_tensor = f.gelu(z_tensor)
    result = result_tensor.data
    expected = (
        0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))
    )
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_gelu_forward_matrix():
    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z)
    result_tensor = f.gelu(z_tensor)
    result = result_tensor.data
    expected = (
        0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))
    )
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_celu_forward_scalar():
    alpha = 1.0
    z = np.random.randn()
    z_tensor = nura.tensor(z)
    result_tensor = f.celu(z_tensor, alpha=alpha)
    result = result_tensor.data
    expected = np.maximum(0, z) + np.minimum(0, alpha * (np.exp(z / alpha) - 1))
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_celu_forward_vector():
    alpha = 1.0
    z = np.random.randn(5)
    z_tensor = nura.tensor(z)
    result_tensor = f.celu(z_tensor, alpha=alpha)
    result = result_tensor.data
    expected = np.maximum(0, z) + np.minimum(0, alpha * (np.exp(z / alpha) - 1))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_celu_forward_matrix():
    alpha = 1.0
    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z)
    result_tensor = f.celu(z_tensor, alpha=alpha)
    result = result_tensor.data
    expected = np.maximum(0, z) + np.minimum(0, alpha * (np.exp(z / alpha) - 1))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_celu_forward_tensor():
    alpha = 1.0
    z = np.random.randn(2, 3, 4)
    z_tensor = nura.tensor(z)
    result_tensor = f.celu(z_tensor, alpha=alpha)
    result = result_tensor.data
    expected = np.maximum(0, z) + np.minimum(0, alpha * (np.exp(z / alpha) - 1))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_celu_forward_alpha():
    alpha = 0.5
    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z)
    result_tensor = f.celu(z_tensor, alpha=alpha)
    result = result_tensor.data
    expected = np.maximum(0, z) + np.minimum(0, alpha * (np.exp(z / alpha) - 1))
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


def test_softmax_forward_tensor_rank4_zero_dim():
    z = np.random.randn(4, 4, 4, 4)

    z_tensor = nura.tensor(z)
    result_tensor = f.softmax(z_tensor, dim=0)
    result = result_tensor.data
    expected = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
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


def test_attention_basic():
    q = np.random.rand(2, 4, 5)
    k = q.copy()
    v = np.random.rand(2, 4, 6)

    context, attn = f.attention(nura.tensor(q), nura.tensor(k), nura.tensor(v))

    dk = q.shape[-1]
    simscore = np.matmul(q, k.transpose(0, 2, 1)) / (dk**0.5)
    attn_expected = np.exp(simscore) / np.sum(np.exp(simscore), axis=-1, keepdims=True)
    context_expected = np.matmul(attn_expected, v)

    np.testing.assert_array_almost_equal(attn.data, attn_expected, decimal=5)
    np.testing.assert_array_almost_equal(context.data, context_expected, decimal=5)


def test_attention_with_mask():
    q = np.random.rand(2, 4, 5)
    k = q.copy()
    v = np.random.rand(2, 4, 6)
    mask = np.tril(np.ones((1, 4, 4)), k=0).astype(bool)

    context, attn = f.attention(
        nura.tensor(q), nura.tensor(k), nura.tensor(v), mask=nura.tensor(mask)
    )

    dk = q.shape[-1]
    simscore = np.matmul(q, k.transpose(0, 2, 1)) / (dk**0.5)
    maskfill = -1e9
    simscore = np.where(mask == True, simscore, maskfill)
    attn_expected = np.exp(simscore) / np.sum(np.exp(simscore), axis=-1, keepdims=True)
    context_expected = np.matmul(attn_expected, v)

    np.testing.assert_array_almost_equal(attn.data, attn_expected, decimal=5)
    np.testing.assert_array_almost_equal(context.data, context_expected, decimal=5)


def test_attention_with_mask_batch():
    q = np.random.rand(2, 4, 5, 3)
    k = q.copy()
    v = np.random.rand(2, 4, 5, 4)
    mask = np.tril(np.ones((1, 5, 5)), k=0).astype(bool)

    context, attn = f.attention(
        nura.tensor(q), nura.tensor(k), nura.tensor(v), mask=nura.tensor(mask)
    )

    dk = q.shape[-1]
    simscore = np.matmul(q, k.transpose(0, 1, 3, 2)) / (dk**0.5)
    maskfill = -1e9
    simscore = np.where(mask == True, simscore, maskfill)
    attn_expected = np.exp(simscore) / np.sum(np.exp(simscore), axis=-1, keepdims=True)
    context_expected = np.matmul(attn_expected, v)

    np.testing.assert_array_almost_equal(attn.data, attn_expected, decimal=5)
    np.testing.assert_array_almost_equal(context.data, context_expected, decimal=5)


def test_embedding_forward_single_index():
    vocab_size = 10
    embedding_dim = 5
    x = np.random.randint(0, vocab_size)
    w = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    x_tensor = nura.tensor(x, dtype=nura.int)
    w_tensor = nura.tensor(w)
    result_tensor = f.embedding(x_tensor, w_tensor)
    result = result_tensor.data
    expected = w[x]
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_embedding_forward_vector():
    vocab_size = 10
    embedding_dim = 5
    seq_length = 3
    x = np.random.randint(0, vocab_size, size=seq_length)
    w = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    x_tensor = nura.tensor(x, dtype=nura.int)
    w_tensor = nura.tensor(w)
    result_tensor = f.embedding(x_tensor, w_tensor)
    result = result_tensor.data
    expected = w[x]
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_embedding_forward_matrix():
    vocab_size = 10
    embedding_dim = 5
    batch_size = 2
    seq_length = 3
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_length))
    w = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    x_tensor = nura.tensor(x, dtype=nura.int)
    w_tensor = nura.tensor(w)
    result_tensor = f.embedding(x_tensor, w_tensor)
    result = result_tensor.data
    expected = w[x]
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_embedding_forward_with_padding():
    vocab_size = 10
    embedding_dim = 5
    batch_size = 2
    seq_length = 3
    padid = 0
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_length))
    x[np.random.rand(*x.shape) < 0.2] = padid
    w = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    x_tensor = nura.tensor(x, dtype=nura.int)
    w_tensor = nura.tensor(w)
    result_tensor = f.embedding(x_tensor, w_tensor, padid=padid)
    result = result_tensor.data
    expected = w[x]
    expected[x == padid] = 0
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_crossentropy_forward_matrix():
    def numpy_crossentropy(z, y, ignoreid=None):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

        mask = y != ignoreid
        indices = mask.nonzero()[0]
        classes = y[indices]
        return -np.mean(np.log(probs[indices, classes]))

    z = np.random.randn(3, 5).astype(np.float32)
    y = np.random.randint(0, 5, size=3).astype(np.int32)
    z_tensor = nura.tensor(z, dtype=nura.float)
    y_tensor = nura.tensor(y, dtype=nura.int)
    result_tensor = f.crossentropy(z_tensor, y_tensor)
    result = result_tensor.data
    expected = numpy_crossentropy(z, y)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_crossentropy_forward_matrix_ignoreid():
    def numpy_crossentropy(z, y, ignoreid=None):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)

        mask = y != ignoreid
        indices = mask.nonzero()[0]
        classes = y[indices]
        return -np.mean(np.log(probs[indices, classes]))

    z = np.random.randn(3, 100).astype(np.float32)
    y = np.random.randint(0, 100, size=3).astype(np.int32)
    ignoreid = 0
    y[np.random.randint(0, 3)] = ignoreid
    z_tensor = nura.tensor(z, dtype=nura.float)
    y_tensor = nura.tensor(y, dtype=nura.int)
    result_tensor = f.crossentropy(z_tensor, y_tensor, ignoreid=ignoreid)
    result = result_tensor.data
    expected = numpy_crossentropy(z, y, ignoreid=ignoreid)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_dropout_forward_scalar():
    z = np.random.randn()
    z_tensor = nura.tensor(z)
    p = 0.5
    result_tensor = f.dropout(z_tensor, p)
    result = result_tensor.data
    assert result.shape == ()


def test_dropout_forward_vector():
    z = np.random.randn(5)
    z_tensor = nura.tensor(z)
    p = 0.1
    result_tensor = f.dropout(z_tensor, p)
    result = result_tensor.data
    assert result.shape == (5,)


def test_dropout_forward_matrix():
    z = np.random.randn(3, 3)
    z_tensor = nura.tensor(z)
    p = 0.9
    result_tensor = f.dropout(z_tensor, p)
    result = result_tensor.data
    assert result.shape == (3, 3)


def test_dropout_forward_tensor():
    z = np.random.randn(2, 3, 4)
    z_tensor = nura.tensor(z)
    p = 0.3
    result_tensor = f.dropout(z_tensor, p)
    result = result_tensor.data
    assert result.shape == (2, 3, 4)


def numpy_binarycrossentropy(x, y):
    epsilon = 1e-7
    x = np.clip(x, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))


def test_binarycrossentropy_scalar():
    x = np.random.rand()
    y = np.random.randint(0, 2)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result = f.binarycrossentropy(x_tensor, y_tensor).data
    expected = numpy_binarycrossentropy(x, y)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_binarycrossentropy_vector():
    x = np.random.rand(5)
    y = np.random.randint(0, 2, size=5)
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result = f.binarycrossentropy(x_tensor, y_tensor).data
    expected = numpy_binarycrossentropy(x, y)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_binarycrossentropy_matrix():
    x = np.random.rand(3, 4)
    y = np.random.randint(0, 2, size=(3, 4))
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result = f.binarycrossentropy(x_tensor, y_tensor).data
    expected = numpy_binarycrossentropy(x, y)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_binarycrossentropy_tensor():
    x = np.random.rand(2, 3, 4)
    y = np.random.randint(0, 2, size=(2, 3, 4))
    x_tensor = nura.tensor(x)
    y_tensor = nura.tensor(y)
    result = f.binarycrossentropy(x_tensor, y_tensor).data
    expected = numpy_binarycrossentropy(x, y)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def numpy_layernorm(x, gamma, beta, dim, correction, eps):
    mean = np.mean(x, axis=dim, keepdims=True)
    var = np.var(x, axis=dim, keepdims=True, ddof=correction)
    std = np.sqrt(var + eps)
    normalized = (x - mean) / std
    return gamma * normalized + beta


def test_layernorm_vector():
    x = np.random.randn(5)
    gamma = np.random.randn(5)
    beta = np.random.randn(5)
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result = f.layernorm(x_tensor, gamma_tensor, beta_tensor, dim=0, eps=1e-5).data
    expected = numpy_layernorm(x, gamma, beta, dim=0, correction=1, eps=1e-5)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_layernorm_matrix():
    x = np.random.randn(3, 4)
    gamma = np.random.randn(4)
    beta = np.random.randn(4)
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result = f.layernorm(x_tensor, gamma_tensor, beta_tensor, dim=-1, eps=1e-5).data
    expected = numpy_layernorm(x, gamma, beta, dim=-1, correction=1, eps=1e-5)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_layernorm_tensor():
    x = np.random.randn(2, 3, 4)
    gamma = np.random.randn(2, 1, 4)
    beta = np.random.randn(2, 1, 4)
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result = f.layernorm(x_tensor, gamma_tensor, beta_tensor, dim=-1, eps=1e-5).data
    expected = numpy_layernorm(x, gamma, beta, dim=-1, correction=1, eps=1e-5)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_layernorm_biased():
    x = np.random.randn(3, 4)
    gamma = np.random.randn(4)
    beta = np.random.randn(4)
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result = f.layernorm(
        x_tensor, gamma_tensor, beta_tensor, dim=-1, correction=0, eps=1e-5
    ).data
    expected = numpy_layernorm(x, gamma, beta, dim=-1, correction=0, eps=1e-5)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_layernorm_unbiased():
    x = np.random.randn(3, 4)
    gamma = np.random.randn(4)
    beta = np.random.randn(4)
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result = f.layernorm(
        x_tensor, gamma_tensor, beta_tensor, dim=-1, correction=1, eps=1e-5
    ).data
    expected = numpy_layernorm(x, gamma, beta, dim=-1, correction=1, eps=1e-5)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_layernorm_aggressive():
    x = np.random.randn(3, 4)
    gamma = np.random.randn(4)
    beta = np.random.randn(4)
    x_tensor = nura.tensor(x)
    gamma_tensor = nura.tensor(gamma)
    beta_tensor = nura.tensor(beta)
    result = f.layernorm(
        x_tensor, gamma_tensor, beta_tensor, dim=-1, correction=2, eps=1e-5
    ).data
    expected = numpy_layernorm(x, gamma, beta, dim=-1, correction=2, eps=1e-5)
    np.testing.assert_almost_equal(result, expected, decimal=5)
