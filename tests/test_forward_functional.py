import numpy as np
import deepnet
import deepnet.functional as f


def test_add_forward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)

    expected = a + b
    np.testing.assert_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_add_forward_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)

    expected = a + b
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_add_forward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)

    expected = a + b
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_sub_forward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)

    expected = a - b
    np.testing.assert_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_sub_forward_vetor():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)

    expected = a - b
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_sub_forward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)

    expected = a - b
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_mul_forward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)

    expected = a * b
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_mul_forward_vetor():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)

    expected = a * b
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_mul_forward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)

    expected = a * b
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_div_forward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)

    expected = a / b
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_div_forward_vetor():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)

    expected = a / b
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_div_forward_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)

    expected = a / b
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_matmul_forward_same_shape():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)

    expected = np.matmul(a, b)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_matmul_forward_different_shape():
    a = np.random.rand(3, 2)
    b = np.random.rand(2, 4)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)

    expected = np.matmul(a, b)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_matmul_forward_rank3_same_shape():
    a = np.random.rand(5, 5, 5)
    b = np.random.rand(5, 5, 5)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)

    expected = np.matmul(a, b)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_matmul_forward_rank3_different_shape():
    a = np.random.rand(3, 4, 5)
    b = np.random.rand(3, 5, 2)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)

    expected = np.matmul(a, b)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_pow_forward_scalar():
    a = np.random.rand()
    b = 2

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    expected = np.power(a, b)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_pow_forward_vector():
    a = np.random.rand(5)
    b = 3

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    expected = np.power(a, b)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_pow_forward_matrix():
    a = np.random.rand(5, 5)
    b = 4

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    expected = np.power(a, b)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_pow_forward_vector_exp():
    a = np.random.rand(4)
    b = np.full_like(a, 2)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    expected = np.power(a, b)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_pow_forward_matrix_exp():
    a = np.random.rand(3, 3)
    b = np.full_like(a, 3)

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)

    expected = np.power(a, b)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)

def test_exp_forward_scalar():
    a = np.random.rand()

    a_tensor = deepnet.tensor(a)
    result_tensor = f.exp(a_tensor)

    expected = np.exp(a)
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)

def test_exp_forward_vector():
    a = np.random.rand(4)

    a_tensor = deepnet.tensor(a)
    result_tensor = f.exp(a_tensor)

    expected = np.exp(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)

def test_exp_forward_matrix():
    a = np.random.rand(2, 3)

    a_tensor = deepnet.tensor(a)
    result_tensor = f.exp(a_tensor)

    expected = np.exp(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)

def test_log_forward_scalar():
    a = np.random.rand()

    a_tensor = deepnet.tensor(a)
    result_tensor = f.log(a_tensor)

    expected = np.log(a)
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)

def test_log_forward_vector():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a)
    result_tensor = f.log(a_tensor)

    expected = np.log(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)

def test_log_forward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a)
    result_tensor = f.log(a_tensor)

    expected = np.log(a)
    np.testing.assert_array_almost_equal(result_tensor.data, expected, decimal=5)


def test_sine_forward_scalar():
    a = np.random.rand()

    a_tensor = deepnet.tensor(a)
    result_tensor = f.sine(a_tensor)

    expected = np.sin(a)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_sine_forward_vector():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a)
    result_tensor = f.sine(a_tensor)

    expected = np.sin(a)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_sine_forward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a)
    result_tensor = f.sine(a_tensor)

    expected = np.sin(a)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_cosine_forward_scalar():
    a = np.random.rand()

    a_tensor = deepnet.tensor(a)
    result_tensor = f.cosine(a_tensor)

    expected = np.cos(a)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_cosine_forward_vector():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a)
    result_tensor = f.cosine(a_tensor)

    expected = np.cos(a)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_cosine_forward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a)
    result_tensor = f.cosine(a_tensor)

    expected = np.cos(a)
    np.testing.assert_array_almost_equal(
        result_tensor.data, expected, decimal=5)


def test_sum_forward_single_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.sum(a_tensor, 1)
    expected = np.sum(a, axis=1)
    assert result_tensor.dim() == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_sum_forward_multiple_dims():
    a = np.random.rand(4, 5, 6)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.sum(a_tensor, (0, 2))
    expected = np.sum(a, axis=(0, 2))
    assert result_tensor.dim() == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_sum_forward_keepdims_true():
    a = np.random.rand(2, 3, 4)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.sum(a_tensor, 1, keepdims=True)
    expected = np.sum(a, axis=1, keepdims=True)
    assert result_tensor.dim() == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_sum_forward_keepdims_false():
    a = np.random.rand(2, 3, 4)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.sum(a_tensor, 1, keepdims=False)
    expected = np.sum(a, axis=1, keepdims=False)
    assert result_tensor.dim() == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_sum_forward_single_element_tensor():
    a = np.random.rand(1)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.sum(a_tensor, 0)
    expected = np.sum(a, axis=0)
    assert result_tensor.dim() == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_sum_forward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.sum(a_tensor, (1, 2))
    expected = np.sum(a, axis=(1, 2))
    assert result_tensor.dim() == expected.shape
    assert np.allclose(result_tensor.data, expected)


def test_squeeze_forward_rank1_v0():
    a = np.random.rand(1)

    a_tensors = deepnet.tensor(a)
    result_tensor = deepnet.squeeze(a_tensors)
    assert result_tensor.dim() == ()


def test_squeeze_forward_rank1_v1():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.squeeze(a_tensor)
    assert result_tensor.dim() == (5,)


def test_squeeze_forward_rank2_v0():
    a = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.squeeze(a_tensor)
    assert result_tensor.dim() == (5, 5)


def test_squeeze_forward_rank2_v1():
    a = np.random.rand(3, 1)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.squeeze(a_tensor)
    assert result_tensor.dim() == (3,)


def test_squeeze_forward_mutli_v0():
    a = np.random.rand(3, 1, 5, 2, 1, 3)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.squeeze(a_tensor)
    assert result_tensor.dim() == (3, 5, 2, 3)


def test_squeeze_forward_multi_v1():
    a = np.random.rand(1, 1, 1, 1, 1, 1, 1, 69, 1)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.squeeze(a_tensor)
    assert result_tensor.dim() == (69,)


def test_squeeze_forward_multi_v2():
    a = np.random.rand(4, 4, 5, 6, 2)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.squeeze(a_tensor)
    assert result_tensor.dim() == (4, 4, 5, 6, 2)


def test_unsqueeze_forward_rank1_v0():
    a = np.random.rand(3)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.unsqueeze(a_tensor, (0, 1))
    assert result_tensor.dim() == (1, 1, 3)


def test_unsqueeze_forward_rank1_v1():
    a = np.random.rand(4)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.unsqueeze(a_tensor, (1, 2))
    assert result_tensor.dim() == (4, 1, 1)


def test_unsqueeze_forward_rank1_v2():
    a = np.random.rand(7)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.unsqueeze(a_tensor, (2, 1, 0))
    assert result_tensor.dim() == (1, 1, 1, 7)


def test_unsqueeze_forward_rank2_v0():
    a = np.random.rand(7, 8)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.unsqueeze(a_tensor, (0))
    assert result_tensor.dim() == (1, 7, 8)


def test_unsqueeze_forward_rank2_v1():
    a = np.random.rand(9, 3)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.unsqueeze(a_tensor, (0, 3))
    assert result_tensor.dim() == (1, 9, 3, 1)


def test_unsqueeze_forward_rank2_v2():
    a = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.unsqueeze(a_tensor, (0, 2, 3))
    assert result_tensor.dim() == (1, 5, 1, 1, 5)


def test_unsqueeze_forward_multi_v0():
    a = np.random.rand(3, 4, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.unsqueeze(a_tensor, (0, 2))
    assert result_tensor.dim() == (1, 3, 1, 4, 5)


def test_unsqueeze_forward_multi_v1():
    a = np.random.rand(2, 3)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.unsqueeze(a_tensor, (1, 3, 4))
    assert result_tensor.dim() == (2, 1, 3, 1, 1)


def test_unsqueeze_forward_multi_v2():
    a = np.random.rand(5, 6, 7, 8)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.unsqueeze(a_tensor, (0, 2, 5))
    assert result_tensor.dim() == (1, 5, 1, 6, 7, 1, 8)


def test_transpose_forward_rank2_v0():
    a = np.random.rand(3, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.transpose(a_tensor)
    assert result_tensor.dim() == (5, 3)


def test_transpose_forward_rank2_v1():
    a = np.random.rand(3, 1)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.transpose(a_tensor, -1, -2)
    assert result_tensor.dim() == (1, 3)


def test_transpose_forward_multi_v0():
    a = np.random.rand(4, 3, 2)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.transpose(a_tensor, -3, -1)
    assert result_tensor.dim() == (2, 3, 4)


def test_transpose_forward_multi_v1():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.transpose(a_tensor, 2, 3)
    assert result_tensor.dim() == (2, 3, 5, 4)


def test_transpose_forward_multi_v2():
    a = np.random.rand(3, 4, 5, 6)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.transpose(a_tensor, 0, 3)
    assert result_tensor.dim() == (6, 4, 5, 3)


def test_permute_forward_rank3_v0():
    a = np.random.rand(64, 10, 512)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.permute(a_tensor, (2, 1, 0))
    assert result_tensor.dim() == (512, 10, 64)


def test_permute_forward_rank3_v1():
    a = np.random.rand(3, 4, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.permute(a_tensor, (1, 0, 2))
    assert result_tensor.dim() == (4, 3, 5)


def test_permute_forward_rank4_v0():
    a = np.random.rand(2, 3, 4, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.permute(a_tensor, (3, 2, 1, 0))
    assert result_tensor.dim() == (5, 4, 3, 2)


def test_permute_forward_rank4_v1():
    a = np.random.rand(5, 6, 7, 8)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.permute(a_tensor, (0, 3, 2, 1))
    assert result_tensor.dim() == (5, 8, 7, 6)


def test_permute_forward_rank2_v0():
    a = np.random.rand(10, 20)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.permute(a_tensor, (1, 0))
    assert result_tensor.dim() == (20, 10)


def test_permute_forward_rank5_v0():
    a = np.random.rand(1, 2, 3, 4, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.permute(a_tensor, (4, 3, 2, 1, 0))
    assert result_tensor.dim() == (5, 4, 3, 2, 1)


def test_view_forward_rank1_to_rank2():
    a = np.random.rand(12)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.view(a_tensor, (4, 3))
    assert result_tensor.dim() == (4, 3)


def test_view_forward_rank2_to_rank1():
    a = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.view(a_tensor, (25,))
    assert result_tensor.dim() == (25,)


def test_view_forward_rank2_to_rank3():
    a = np.random.rand(8, 6)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.view(a_tensor, (2, 4, 6))
    assert result_tensor.dim() == (2, 4, 6)


def test_view_forward_rank3_to_rank2():
    a = np.random.rand(3, 4, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.view(a_tensor, (12, 5))
    assert result_tensor.dim() == (12, 5)


def test_view_forward_rank3_to_rank4():
    a = np.random.rand(3, 2, 6)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.view(a_tensor, (3, 1, 2, 6))
    assert result_tensor.dim() == (3, 1, 2, 6)


def test_view_forward_rank4_to_rank2():
    a = np.random.rand(2, 3, 4, 2)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.view(a_tensor, (6, 8))
    assert result_tensor.dim() == (6, 8)


def test_view_forward_with_negative_dim():
    a = np.random.rand(4, 3, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.view(a_tensor, (-1, 6))
    assert result_tensor.dim() == (10, 6)


def test_reshape_forward_rank1_to_rank2():
    a = np.random.rand(10)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.reshape(a_tensor, (5, 2))
    assert result_tensor.dim() == (5, 2)


def test_reshape_forward_rank2_to_rank1():
    a = np.random.rand(4, 3)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.reshape(a_tensor, (12,))
    assert result_tensor.dim() == (12,)


def test_reshape_forward_rank2_to_rank3():
    a = np.random.rand(6, 4)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.reshape(a_tensor, (2, 3, 4))
    assert result_tensor.dim() == (2, 3, 4)


def test_reshape_forward_rank3_to_rank2():
    a = np.random.rand(2, 3, 4)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.reshape(a_tensor, (6, 4))
    assert result_tensor.dim() == (6, 4)


def test_reshape_forward_rank3_to_rank4():
    a = np.random.rand(2, 3, 4)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.reshape(a_tensor, (2, 2, 3, 2))
    assert result_tensor.dim() == (2, 2, 3, 2)


def test_reshape_forward_rank4_to_rank2():
    a = np.random.rand(2, 2, 3, 2)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.reshape(a_tensor, (4, 6))
    assert result_tensor.dim() == (4, 6)


def test_reshape_forward_with_negative_dim():
    a = np.random.rand(3, 4, 5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.reshape(a_tensor, (-1, 5))
    assert result_tensor.dim() == (12, 5)


def test_clone_forward_scalar():
    a = np.random.rand()

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.clone(a_tensor)

    assert np.allclose(result_tensor.data, a_tensor.data)
    assert result_tensor.data is not a_tensor.data


def test_clone_forward_vector():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.clone(a_tensor)

    assert np.allclose(result_tensor.data, a_tensor.data)
    assert result_tensor.data is not a_tensor.data


def test_clone_forward_matrix():
    a = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.clone(a_tensor)

    assert np.allclose(result_tensor.data, a_tensor.data)
    assert result_tensor.data is not a_tensor.data


def test_clone_forward_higher_rank_tensor():
    a = np.random.rand(2, 3, 4)

    a_tensor = deepnet.tensor(a)
    result_tensor = deepnet.clone(a_tensor)

    assert np.allclose(result_tensor.data, a_tensor.data)
    assert result_tensor.data is not a_tensor.data


def main():

    # Add Tests

    test_add_forward_scalar()
    test_add_forward_vector()
    test_add_forward_matrix()

    # Sub Tests

    test_sub_forward_scalar()
    test_sub_forward_vetor()
    test_sub_forward_matrix()

    # Mul Tests

    test_mul_forward_scalar()
    test_mul_forward_vetor()
    test_mul_forward_matrix()

    # Div Tests

    test_div_forward_scalar()
    test_div_forward_vetor()
    test_div_forward_matrix()

    # Matmul Tests

    test_matmul_forward_same_shape()
    test_matmul_forward_different_shape()
    test_matmul_forward_rank3_same_shape()
    test_matmul_forward_rank3_different_shape()

    # Pow Tests

    test_pow_forward_scalar()
    test_pow_forward_vector()
    test_pow_forward_matrix()
    test_pow_forward_vector_exp()
    test_pow_forward_matrix_exp()

    # Exp Tests

    test_exp_forward_scalar()
    test_exp_forward_vector()
    test_exp_forward_matrix()

    # Log Tests

    test_log_forward_scalar()
    test_log_forward_vector()
    test_log_forward_matrix()

    # Sine Tests

    test_sine_forward_scalar()
    test_sine_forward_vector()
    test_sine_forward_matrix()

    # Cosine Tests

    test_cosine_forward_scalar()
    test_cosine_forward_vector()
    test_cosine_forward_matrix()

    # Sum Tests

    test_sum_forward_single_dim()
    test_sum_forward_multiple_dims()
    test_sum_forward_higher_rank_tensor()

    test_sum_forward_single_element_tensor()
    test_sum_forward_keepdims_false()
    test_sum_forward_keepdims_true()

    # Squeeze Tests

    test_squeeze_forward_rank1_v0()
    test_squeeze_forward_rank1_v1()
    test_squeeze_forward_rank2_v0()
    test_squeeze_forward_rank2_v1()

    test_squeeze_forward_mutli_v0()
    test_squeeze_forward_multi_v1()
    test_squeeze_forward_multi_v2()

    # Unsqueeze Tests

    test_unsqueeze_forward_rank1_v0()
    test_unsqueeze_forward_rank1_v1()
    test_unsqueeze_forward_rank1_v2()

    test_unsqueeze_forward_rank2_v0()
    test_unsqueeze_forward_rank2_v1()
    test_unsqueeze_forward_rank2_v2()

    test_unsqueeze_forward_multi_v0()
    test_unsqueeze_forward_multi_v1()
    test_unsqueeze_forward_multi_v2()

    # Transpose Tests

    test_transpose_forward_rank2_v0()
    test_transpose_forward_rank2_v1()

    test_transpose_forward_multi_v0()
    test_transpose_forward_multi_v1()
    test_transpose_forward_multi_v2()

    # Permute Tests

    test_permute_forward_rank2_v0()
    test_permute_forward_rank3_v0()
    test_permute_forward_rank3_v1()
    test_permute_forward_rank4_v0()
    test_permute_forward_rank4_v1()
    test_permute_forward_rank5_v0()

    # View Tests

    test_view_forward_rank1_to_rank2()
    test_view_forward_rank2_to_rank3()
    test_view_forward_rank3_to_rank4()

    test_view_forward_rank2_to_rank1()
    test_view_forward_rank3_to_rank2()
    test_view_forward_rank4_to_rank2()
    test_view_forward_with_negative_dim()

    # Reshape Tests

    test_reshape_forward_rank1_to_rank2()
    test_reshape_forward_rank2_to_rank3()
    test_reshape_forward_rank3_to_rank4()

    test_reshape_forward_rank2_to_rank1()
    test_reshape_forward_rank3_to_rank2()
    test_reshape_forward_rank4_to_rank2()
    test_reshape_forward_with_negative_dim()

    # Clone Tests

    test_clone_forward_scalar()
    test_clone_forward_vector()
    test_clone_forward_matrix()
    test_clone_forward_higher_rank_tensor()

    print("All tests passed")


if __name__ == "__main__":
    main()
