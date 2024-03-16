import numpy as np
import nura
import nura.functional as f


def test_add_forward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)

    expected = a + b
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)


def test_add_forward_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)

    expected = a + b
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_add_forward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)

    expected = a + b
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_sub_forward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)

    expected = a - b
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)


def test_sub_forward_vetor():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)

    expected = a - b
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_sub_forward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)

    expected = a - b
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_mul_forward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)

    expected = a * b
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_mul_forward_vetor():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)

    expected = a * b
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_mul_forward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)

    expected = a * b
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_div_forward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)

    expected = a / b
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_div_forward_vetor():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)

    expected = a / b
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_div_forward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)

    expected = a / b
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_dot_forward_vectors():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.dot(a_tensor, b_tensor)

    expected = np.dot(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_dot_forward_matrix_vector():
    a = np.random.rand(3, 5)
    b = np.random.rand(5)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.dot(a_tensor, b_tensor)

    expected = np.dot(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_dot_forward_matrix_matrix():
    a = np.random.rand(3, 4)
    b = np.random.rand(4, 2)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.dot(a_tensor, b_tensor)

    expected = np.dot(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_matmul_forward_same_shape():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)

    expected = np.matmul(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_matmul_forward_different_shape():
    a = np.random.rand(3, 2)
    b = np.random.rand(2, 4)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)

    expected = np.matmul(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_matmul_forward_rank3_same_shape():
    a = np.random.rand(5, 5, 5)
    b = np.random.rand(5, 5, 5)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)

    expected = np.matmul(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_matmul_forward_rank3_different_shape():
    a = np.random.rand(3, 4, 5)
    b = np.random.rand(3, 5, 2)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)

    expected = np.matmul(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_pow_forward_scalar():
    a = np.random.rand()
    b = 2

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    expected = np.power(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_pow_forward_vector():
    a = np.random.rand(5)
    b = 3

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    expected = np.power(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_pow_forward_matrix():
    a = np.random.rand(5, 5)
    b = 4

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    expected = np.power(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_pow_forward_vector_exp():
    a = np.random.rand(4)
    b = np.full_like(a, 2)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    expected = np.power(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_pow_forward_matrix_exp():
    a = np.random.rand(3, 3)
    b = np.full_like(a, 3)

    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    expected = np.power(a, b)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_square_scalar():
    a = np.random.rand()
    a_tensor = nura.tensor(a)
    result_tensor = f.square(a_tensor)
    expected = a**2
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)


def test_square_vector():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a)
    result_tensor = f.square(a_tensor)
    expected = np.square(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_square_matrix():
    a = np.random.rand(3, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.square(a_tensor)
    expected = np.square(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_sqrt_scalar():
    a = np.random.rand()
    a_tensor = nura.tensor(a)
    result_tensor = f.sqrt(a_tensor)
    expected = np.sqrt(a)
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)


def test_sqrt_vector():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a)
    result_tensor = f.sqrt(a_tensor)
    expected = np.sqrt(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_sqrt_matrix():
    a = np.random.rand(3, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.sqrt(a_tensor)
    expected = np.sqrt(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_exp_forward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a)
    result_tensor = f.exp(a_tensor)

    expected = np.exp(a)
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)


def test_exp_forward_vector():
    a = np.random.rand(4)

    a_tensor = nura.tensor(a)
    result_tensor = f.exp(a_tensor)

    expected = np.exp(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_exp_forward_matrix():
    a = np.random.rand(2, 3)

    a_tensor = nura.tensor(a)
    result_tensor = f.exp(a_tensor)

    expected = np.exp(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_log_forward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a)
    result_tensor = f.log(a_tensor)

    expected = np.log(a)
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)


def test_log_forward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a)
    result_tensor = f.log(a_tensor)

    expected = np.log(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_log_forward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = nura.tensor(a)
    result_tensor = f.log(a_tensor)

    expected = np.log(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_sin_forward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a)
    result_tensor = f.sin(a_tensor)

    expected = np.sin(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_sin_forward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a)
    result_tensor = f.sin(a_tensor)

    expected = np.sin(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_sin_forward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = nura.tensor(a)
    result_tensor = f.sin(a_tensor)

    expected = np.sin(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_cos_forward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a)
    result_tensor = f.cos(a_tensor)

    expected = np.cos(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_cos_forward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a)
    result_tensor = f.cos(a_tensor)

    expected = np.cos(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_cos_forward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = nura.tensor(a)
    result_tensor = f.cos(a_tensor)

    expected = np.cos(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_sum_forward_single_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.sum(a_tensor, 1)
    expected = np.sum(a, axis=1)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_sum_forward_multiple_dims():
    a = np.random.rand(4, 5, 6)

    a_tensor = nura.tensor(a)
    result_tensor = nura.sum(a_tensor, (0, 2))
    expected = np.sum(a, axis=(0, 2))
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_sum_forward_keepdims_true():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a)
    result_tensor = nura.sum(a_tensor, 1, keepdims=True)
    expected = np.sum(a, axis=1, keepdims=True)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_sum_forward_keepdims_false():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a)
    result_tensor = nura.sum(a_tensor, 1, keepdims=False)
    expected = np.sum(a, axis=1, keepdims=False)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_sum_forward_single_element_tensor():
    a = np.random.rand(1)

    a_tensor = nura.tensor(a)
    result_tensor = nura.sum(a_tensor, 0)
    expected = np.sum(a, axis=0)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_sum_forward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.sum(a_tensor, (1, 2))
    expected = np.sum(a, axis=(1, 2))
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_max_forward_single_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, 1)
    expected = np.max(a, axis=1)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_max_forward_multiple_dims():
    a = np.random.rand(4, 5, 6)

    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, (0, 2))
    expected = np.max(a, axis=(0, 2))
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_max_forward_keepdims_true():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, 1, keepdims=True)
    expected = np.max(a, axis=1, keepdims=True)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_max_forward_keepdims_false():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, 1, keepdims=False)
    expected = np.max(a, axis=1, keepdims=False)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_max_forward_single_element_tensor():
    a = np.random.rand(1)

    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, 0)
    expected = np.max(a, axis=0)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_max_forward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, (1, 2))
    expected = np.max(a, axis=(1, 2))
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_min_forward_single_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, 1)
    expected = np.min(a, axis=1)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_min_forward_multiple_dims():
    a = np.random.rand(4, 5, 6)

    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, (0, 2))
    expected = np.min(a, axis=(0, 2))
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_min_forward_keepdims_true():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, 1, keepdims=True)
    expected = np.min(a, axis=1, keepdims=True)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_min_forward_keepdims_false():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, 1, keepdims=False)
    expected = np.min(a, axis=1, keepdims=False)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_min_forward_single_element_tensor():
    a = np.random.rand(1)

    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, 0)
    expected = np.min(a, axis=0)
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_min_forward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, (1, 2))
    expected = np.min(a, axis=(1, 2))
    assert result_tensor.dim == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_abs_forward_scalar():
    a = np.random.rand() * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a)
    result_tensor = f.abs(a_tensor)
    expected = np.absolute(a)
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)


def test_abs_forward_vector():
    a = np.random.rand(5) * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a)
    result_tensor = f.abs(a_tensor)
    expected = np.absolute(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_abs_forward_matrix():
    a = np.random.rand(3, 3) * np.random.choice([-1, 1], size=(3, 3))

    a_tensor = nura.tensor(a)
    result_tensor = f.abs(a_tensor)
    expected = np.absolute(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_pos_forward_scalar():
    a = np.random.rand() * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a)
    result_tensor = f.pos(a_tensor)
    expected = np.positive(a)
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)


def test_pos_forward_vector():
    a = np.random.rand(5) * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a)
    result_tensor = f.pos(a_tensor)
    expected = np.positive(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_pos_forward_matrix():
    a = np.random.rand(3, 3) * np.random.choice([-1, 1], size=(3, 3))

    a_tensor = nura.tensor(a)
    result_tensor = f.pos(a_tensor)
    expected = np.positive(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_neg_forward_scalar():
    a = np.random.rand() * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a)
    result_tensor = f.neg(a_tensor)
    expected = np.negative(a)
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)


def test_neg_forward_vector():
    a = np.random.rand(5) * np.random.choice([-1, 1])

    a_tensor = nura.tensor(a)
    result_tensor = f.neg(a_tensor)
    expected = np.negative(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_neg_forward_matrix():
    a = np.random.rand(3, 3) * np.random.choice([-1, 1], size=(3, 3))
    a_tensor = nura.tensor(a)
    result_tensor = f.neg(a_tensor)
    expected = np.negative(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_squeeze_forward_rank1_v0():
    a = np.random.rand(1)

    a_tensors = nura.tensor(a)
    result_tensor = nura.squeeze(a_tensors)
    assert result_tensor.dim == ()


def test_squeeze_forward_rank1_v1():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.squeeze(a_tensor)
    assert result_tensor.dim == (5,)


def test_squeeze_forward_rank2_v0():
    a = np.random.rand(5, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.squeeze(a_tensor)
    assert result_tensor.dim == (5, 5)


def test_squeeze_forward_rank2_v1():
    a = np.random.rand(3, 1)

    a_tensor = nura.tensor(a)
    result_tensor = nura.squeeze(a_tensor)
    assert result_tensor.dim == (3,)


def test_squeeze_forward_mutli_v0():
    a = np.random.rand(3, 1, 5, 2, 1, 3)

    a_tensor = nura.tensor(a)
    result_tensor = nura.squeeze(a_tensor)
    assert result_tensor.dim == (3, 5, 2, 3)


def test_squeeze_forward_multi_v1():
    a = np.random.rand(1, 1, 1, 1, 1, 1, 1, 69, 1)

    a_tensor = nura.tensor(a)
    result_tensor = nura.squeeze(a_tensor)
    assert result_tensor.dim == (69,)


def test_squeeze_forward_multi_v2():
    a = np.random.rand(4, 4, 5, 6, 2)

    a_tensor = nura.tensor(a)
    result_tensor = nura.squeeze(a_tensor)
    assert result_tensor.dim == (4, 4, 5, 6, 2)


def test_unsqueeze_forward_rank1_v0():
    a = np.random.rand(3)

    a_tensor = nura.tensor(a)
    result_tensor = nura.unsqueeze(a_tensor, (0, 1))
    assert result_tensor.dim == (1, 1, 3)


def test_unsqueeze_forward_rank1_v1():
    a = np.random.rand(4)

    a_tensor = nura.tensor(a)
    result_tensor = nura.unsqueeze(a_tensor, (1, 2))
    assert result_tensor.dim == (4, 1, 1)


def test_unsqueeze_forward_rank1_v2():
    a = np.random.rand(7)

    a_tensor = nura.tensor(a)
    result_tensor = nura.unsqueeze(a_tensor, (2, 1, 0))
    assert result_tensor.dim == (1, 1, 1, 7)


def test_unsqueeze_forward_rank2_v0():
    a = np.random.rand(7, 8)

    a_tensor = nura.tensor(a)
    result_tensor = nura.unsqueeze(a_tensor, (0))
    assert result_tensor.dim == (1, 7, 8)


def test_unsqueeze_forward_rank2_v1():
    a = np.random.rand(9, 3)

    a_tensor = nura.tensor(a)
    result_tensor = nura.unsqueeze(a_tensor, (0, 3))
    assert result_tensor.dim == (1, 9, 3, 1)


def test_unsqueeze_forward_rank2_v2():
    a = np.random.rand(5, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.unsqueeze(a_tensor, (0, 2, 3))
    assert result_tensor.dim == (1, 5, 1, 1, 5)


def test_unsqueeze_forward_multi_v0():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.unsqueeze(a_tensor, (0, 2))
    assert result_tensor.dim == (1, 3, 1, 4, 5)


def test_unsqueeze_forward_multi_v1():
    a = np.random.rand(2, 3)

    a_tensor = nura.tensor(a)
    result_tensor = nura.unsqueeze(a_tensor, (1, 3, 4))
    assert result_tensor.dim == (2, 1, 3, 1, 1)


def test_unsqueeze_forward_multi_v2():
    a = np.random.rand(5, 6, 7, 8)

    a_tensor = nura.tensor(a)
    result_tensor = nura.unsqueeze(a_tensor, (0, 2, 5))
    assert result_tensor.dim == (1, 5, 1, 6, 7, 1, 8)


def test_transpose_forward_rank2_v0():
    a = np.random.rand(3, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.transpose(a_tensor)
    assert result_tensor.dim == (5, 3)


def test_transpose_forward_rank2_v1():
    a = np.random.rand(3, 1)

    a_tensor = nura.tensor(a)
    result_tensor = nura.transpose(a_tensor, -1, -2)
    assert result_tensor.dim == (1, 3)


def test_transpose_forward_multi_v0():
    a = np.random.rand(4, 3, 2)

    a_tensor = nura.tensor(a)
    result_tensor = nura.transpose(a_tensor, -3, -1)
    assert result_tensor.dim == (2, 3, 4)


def test_transpose_forward_multi_v1():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.transpose(a_tensor, 2, 3)
    assert result_tensor.dim == (2, 3, 5, 4)


def test_transpose_forward_multi_v2():
    a = np.random.rand(3, 4, 5, 6)

    a_tensor = nura.tensor(a)
    result_tensor = nura.transpose(a_tensor, 0, 3)
    assert result_tensor.dim == (6, 4, 5, 3)


def test_permute_forward_rank3_v0():
    a = np.random.rand(64, 10, 512)

    a_tensor = nura.tensor(a)
    result_tensor = nura.permute(a_tensor, (2, 1, 0))
    assert result_tensor.dim == (512, 10, 64)


def test_permute_forward_rank3_v1():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.permute(a_tensor, (1, 0, 2))
    assert result_tensor.dim == (4, 3, 5)


def test_permute_forward_rank4_v0():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.permute(a_tensor, (3, 2, 1, 0))
    assert result_tensor.dim == (5, 4, 3, 2)


def test_permute_forward_rank4_v1():
    a = np.random.rand(5, 6, 7, 8)

    a_tensor = nura.tensor(a)
    result_tensor = nura.permute(a_tensor, (0, 3, 2, 1))
    assert result_tensor.dim == (5, 8, 7, 6)


def test_permute_forward_rank2_v0():
    a = np.random.rand(10, 20)

    a_tensor = nura.tensor(a)
    result_tensor = nura.permute(a_tensor, (1, 0))
    assert result_tensor.dim == (20, 10)


def test_permute_forward_rank5_v0():
    a = np.random.rand(1, 2, 3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.permute(a_tensor, (4, 3, 2, 1, 0))
    assert result_tensor.dim == (5, 4, 3, 2, 1)


def test_view_forward_rank1_to_rank2():
    a = np.random.rand(12)

    a_tensor = nura.tensor(a)
    result_tensor = nura.view(a_tensor, (4, 3))
    assert result_tensor.dim == (4, 3)


def test_view_forward_rank2_to_rank1():
    a = np.random.rand(5, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.view(a_tensor, (25,))
    assert result_tensor.dim == (25,)


def test_view_forward_rank2_to_rank3():
    a = np.random.rand(8, 6)

    a_tensor = nura.tensor(a)
    result_tensor = nura.view(a_tensor, (2, 4, 6))
    assert result_tensor.dim == (2, 4, 6)


def test_view_forward_rank3_to_rank2():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.view(a_tensor, (12, 5))
    assert result_tensor.dim == (12, 5)


def test_view_forward_rank3_to_rank4():
    a = np.random.rand(3, 2, 6)

    a_tensor = nura.tensor(a)
    result_tensor = nura.view(a_tensor, (3, 1, 2, 6))
    assert result_tensor.dim == (3, 1, 2, 6)


def test_view_forward_rank4_to_rank2():
    a = np.random.rand(2, 3, 4, 2)

    a_tensor = nura.tensor(a)
    result_tensor = nura.view(a_tensor, (6, 8))
    assert result_tensor.dim == (6, 8)


def test_view_forward_with_negative_dim():
    a = np.random.rand(4, 3, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.view(a_tensor, (-1, 6))
    assert result_tensor.dim == (10, 6)


def test_reshape_forward_rank1_to_rank2():
    a = np.random.rand(10)

    a_tensor = nura.tensor(a)
    result_tensor = nura.reshape(a_tensor, (5, 2))
    assert result_tensor.dim == (5, 2)


def test_reshape_forward_rank2_to_rank1():
    a = np.random.rand(4, 3)

    a_tensor = nura.tensor(a)
    result_tensor = nura.reshape(a_tensor, (12,))
    assert result_tensor.dim == (12,)


def test_reshape_forward_rank2_to_rank3():
    a = np.random.rand(6, 4)

    a_tensor = nura.tensor(a)
    result_tensor = nura.reshape(a_tensor, (2, 3, 4))
    assert result_tensor.dim == (2, 3, 4)


def test_reshape_forward_rank3_to_rank2():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a)
    result_tensor = nura.reshape(a_tensor, (6, 4))
    assert result_tensor.dim == (6, 4)


def test_reshape_forward_rank3_to_rank4():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a)
    result_tensor = nura.reshape(a_tensor, (2, 2, 3, 2))
    assert result_tensor.dim == (2, 2, 3, 2)


def test_reshape_forward_rank4_to_rank2():
    a = np.random.rand(2, 2, 3, 2)

    a_tensor = nura.tensor(a)
    result_tensor = nura.reshape(a_tensor, (4, 6))
    assert result_tensor.dim == (4, 6)


def test_reshape_forward_with_negative_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.reshape(a_tensor, (-1, 5))
    assert result_tensor.dim == (12, 5)


def test_clone_forward_scalar():
    a = np.random.rand()

    a_tensor = nura.tensor(a)
    result_tensor = nura.clone(a_tensor)

    assert np.allclose(result_tensor.data, a_tensor.data)
    assert result_tensor.data is not a_tensor.data


def test_clone_forward_vector():
    a = np.random.rand(5)

    a_tensor = nura.tensor(a)
    result_tensor = nura.clone(a_tensor)

    assert np.allclose(result_tensor.data, a_tensor.data)
    assert result_tensor.data is not a_tensor.data


def test_clone_forward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = nura.tensor(a)
    result_tensor = nura.clone(a_tensor)

    assert np.allclose(result_tensor.data, a_tensor.data)
    assert result_tensor.data is not a_tensor.data


def test_clone_forward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4)

    a_tensor = nura.tensor(a)
    result_tensor = nura.clone(a_tensor)

    assert np.allclose(result_tensor.data, a_tensor.data)
    assert result_tensor.data is not a_tensor.data


def test_slice_forward_single_index():
    a = np.random.rand(5, 5)

    a_tensor = nura.tensor(a)
    result_tensor = a_tensor[2, :]

    expected = a[2, :]
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_slice_forward_range():
    a = np.random.rand(10, 10)

    a_tensor = nura.tensor(a)
    result_tensor = a_tensor[2:5, 3:7]

    expected = a[2:5, 3:7]
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_slice_forward_step():
    a = np.random.rand(8, 8)

    a_tensor = nura.tensor(a)
    result_tensor = a_tensor[::2, ::3]

    expected = a[::2, ::3]
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_slice_forward_negative_indices():
    a = np.random.rand(6, 6)

    a_tensor = nura.tensor(a)
    result_tensor = a_tensor[-3:, -3:]

    expected = a[-3:, -3:]
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_slice_forward_mixed_indices():
    a = np.random.rand(7, 7)

    a_tensor = nura.tensor(a)
    result_tensor = a_tensor[1:5, -3]

    expected = a[1:5, -3]
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)
