import numpy as np
import deepnet
import deepnet.functional as f

def test_add_jvp_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.add(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h + b - (a - h + b)) / (2 * h)
    expected_b = (a + b + h - (a + b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_add_jvp_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones(5)))
    with deepnet.forward_ad():
        result_tensor = f.add(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h + b - (a - h + b)) / (2 * h)
    expected_b = (a + b + h - (a + b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_add_jvp_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 3))))
    with deepnet.forward_ad():
        result_tensor = f.add(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h + b - (a - h + b)) / (2 * h)
    expected_b = (a + b + h - (a + b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_sub_jvp_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.sub(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h - b - (a - h - b)) / (2 * h)
    expected_b = (a - (b + h) - (a - (b - h))) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_sub_jvp_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones(5)))
    with deepnet.forward_ad():
        result_tensor = f.sub(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h - b - (a - h - b)) / (2 * h)
    expected_b = (a - (b + h) - (a - (b - h))) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_sub_jvp_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 3))))
    with deepnet.forward_ad():
        result_tensor = f.sub(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h - b - (a - h - b)) / (2 * h)
    expected_b = (a - (b + h) - (a - (b - h))) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_mul_jvp_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.mul(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) * b - (a - h) * b) / (2 * h)
    expected_b = (a * (b + h) - a * (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_mul_jvp_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones(5)))
    with deepnet.forward_ad():
        result_tensor = f.mul(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) * b - (a - h) * b) / (2 * h)
    expected_b = (a * (b + h) - a * (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_mul_jvp_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 3))))
    with deepnet.forward_ad():
        result_tensor = f.mul(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) * b - (a - h) * b) / (2 * h)
    expected_b = (a * (b + h) - a * (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_div_jvp_scalar():
    a = np.random.rand()   
    b = np.random.rand() 

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.div(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_b = (a / (b + h) - a / (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_div_jvp_vector():
    a = np.random.rand(5) 
    b = np.random.rand(5) 

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones(5)))
    with deepnet.forward_ad():
        result_tensor = f.div(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_b = (a / (b + h) - a / (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)


def test_div_jvp_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 3))))
    with deepnet.forward_ad():
        result_tensor = f.div(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_b = (a / (b + h) - a / (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

# Using symbolic differentiaton as numeric differentiation, gives
# unwanted results

def test_matmul_jvp_square_matrices():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 3))))
    with deepnet.forward_ad():
        result_tensor = f.matmul(a_tensor, b_tensor)

    expected_tangent = np.matmul(a, np.ones((3, 3))) + np.matmul(np.ones((3, 3)), b)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_matmul_jvp_different_shapes():
    a = np.random.rand(4, 3)
    b = np.random.rand(3, 5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((4, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 5))))
    with deepnet.forward_ad():
        result_tensor = f.matmul(a_tensor, b_tensor)

    expected_tangent = np.matmul(a, np.ones((3, 5))) + np.matmul(np.ones((4, 3)), b)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)


def test_matmul_jvp_higher_rank_same_shape():
    a = np.random.rand(2, 3, 4, 4)
    b = np.random.rand(2, 3, 4, 4)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 3, 4, 4))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((2, 3, 4, 4))))
    with deepnet.forward_ad():
        result_tensor = f.matmul(a_tensor, b_tensor)

    expected_tangent = np.matmul(a, np.ones((2, 3, 4, 4))) + np.matmul(np.ones((2, 3, 4, 4)), b)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_matmul_jvp_higher_rank_different_shape():
    a = np.random.rand(2, 4, 3)
    b = np.random.rand(2, 3, 5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 4, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((2, 3, 5))))
    with deepnet.forward_ad():
        result_tensor = f.matmul(a_tensor, b_tensor)

    expected_tangent = np.matmul(a, np.ones((2, 3, 5))) + np.matmul(np.ones((2, 4, 3)), b)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_pow_jvp_scalar():
    a = np.random.rand()
    b = 2.

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.pow(a_tensor, b_tensor)

    h = 1e-8
    expected_tangent_a = ((a + h) ** b - (a - h) ** b) / (2 * h)
    expected_tangent_b = (a ** (b + h) - a ** (b - h)) / (2 * h)
    expected_tangent = expected_tangent_a + expected_tangent_b
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_pow_jvp_vector():
    a = np.random.rand(4)
    b = 3.

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(4)))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.pow(a_tensor, b_tensor)

    h = 1e-8
    expected_tangent_a = ((a + h) ** b - (a - h) ** b) / (2 * h)
    expected_tangent_b = (a ** (b + h) - a ** (b - h)) / (2 * h)
    expected_tangent = expected_tangent_a + expected_tangent_b
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_pow_jvp_matrix():
    a = np.random.rand(3, 3)
    b = 4.

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.pow(a_tensor, b_tensor)

    h = 1e-8
    expected_tangent_a = ((a + h) ** b - (a - h) ** b) / (2 * h)
    expected_tangent_b = (a ** (b + h) - a ** (b - h)) / (2 * h)
    expected_tangent = expected_tangent_a + expected_tangent_b
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_pow_jvp_vector_exp():
    a = np.random.rand(4)
    b = np.full_like(a, 2)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(4)))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones(4)))
    with deepnet.forward_ad():
        result_tensor = f.pow(a_tensor, b_tensor)

    h = 1e-8
    expected_tangent_a = ((a + h) ** b - (a - h) ** b) / (2 * h)
    expected_tangent_b = (a ** (b + h) - a ** (b - h)) / (2 * h)
    expected_tangent = expected_tangent_a + expected_tangent_b
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_pow_jvp_matrix_exp():
    a = np.random.rand(3, 3)
    b = np.full_like(a, 3)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 3))))
    with deepnet.forward_ad():
        result_tensor = f.pow(a_tensor, b_tensor)

    h = 1e-8
    expected_tangent_a = ((a + h) ** b - (a - h) ** b) / (2 * h)
    expected_tangent_b = (a ** (b + h) - a ** (b - h)) / (2 * h)
    expected_tangent = expected_tangent_a + expected_tangent_b
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_exp_jvp_scalar():
    a = np.random.rand()

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.exp(a_tensor)

    h = 1e-8
    expected_tangent = (np.exp(a + h) - np.exp(a - h)) / (2 * h)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_exp_jvp_vector():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    with deepnet.forward_ad():
        result_tensor = f.exp(a_tensor)

    h = 1e-8
    expected_tangent = (np.exp(a + h) - np.exp(a - h)) / (2 * h)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_exp_jvp_matrix():
    a = np.random.rand(3, 4)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 4))))
    with deepnet.forward_ad():
        result_tensor = f.exp(a_tensor)

    h = 1e-8
    expected_tangent = (np.exp(a + h) - np.exp(a - h)) / (2 * h)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_log_jvp_scalar():
    a = np.random.rand()

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.log(a_tensor)

    h = 1e-8
    expected_tangent = ((np.log(a + h) - np.log(a - h)) / (2 * h))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_log_jvp_vector():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    with deepnet.forward_ad():
        result_tensor = f.log(a_tensor)

    h = 1e-8
    expected_tangent = ((np.log(a + h) - np.log(a - h)) / (2 * h))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_log_jvp_matrix():
    a = np.random.rand(3, 4)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 4))))
    with deepnet.forward_ad():
        result_tensor = f.log(a_tensor)

    h = 1e-8
    expected_tangent = ((np.log(a + h) - np.log(a - h)) / (2 * h))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_sine_jvp_scalar():
    a = np.random.rand()
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.sine(a_tensor)
    h = 1e-8
    expected_tangent = ((np.sin(a + h) - np.sin(a - h)) / (2 * h))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_sine_jvp_vector():
    a = np.random.rand(4)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(4)))
    with deepnet.forward_ad():
        result_tensor = f.sine(a_tensor)
    h = 1e-8
    expected_tangent = ((np.sin(a + h) - np.sin(a - h)) / (2 * h))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_sine_jvp_matrix():
    a = np.random.rand(3, 3)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    with deepnet.forward_ad():
        result_tensor = f.sine(a_tensor)
    h = 1e-8
    expected_tangent = ((np.sin(a + h) - np.sin(a - h)) / (2 * h))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_cosine_jvp_scalar():
    a = np.random.rand()
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    with deepnet.forward_ad():
        result_tensor = f.cosine(a_tensor)
    h = 1e-8
    expected_tangent = ((np.cos(a + h) - np.cos(a - h)) / (2 * h))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_cosine_jvp_vector():
    a = np.random.rand(4)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(4)))
    with deepnet.forward_ad():
        result_tensor = f.cosine(a_tensor)
    h = 1e-8
    expected_tangent = ((np.cos(a + h) - np.cos(a - h)) / (2 * h))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_cosine_jvp_matrix():
    a = np.random.rand(4, 4)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((4, 4))))
    with deepnet.forward_ad():
        result_tensor = f.cosine(a_tensor)
    h = 1e-8
    expected_tangent = ((np.cos(a + h) - np.cos(a - h)) / (2 * h))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_sum_jvp_single_dim():
    a = np.random.rand(3, 4)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 4))))
    with deepnet.forward_ad():
        result_tensor = deepnet.sum(a_tensor, dims=1)
    expected_tangent = np.sum(np.ones((3, 4)), axis=1)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_sum_jvp_multiple_dim():
    a = np.random.rand(3, 4, 5)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 4, 5))))
    with deepnet.forward_ad():
        result_tensor = deepnet.sum(a_tensor, dims=(1, 2))
    expected_tangent = np.sum(np.ones((3, 4, 5)), axis=(1, 2))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_sum_jvp_keepdims():
    a = np.random.rand(3, 4)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 4))))
    with deepnet.forward_ad():
        result_tensor = deepnet.sum(a_tensor, dims=1, keepdims=True)
    expected_tangent = np.sum(np.ones((3, 4)), axis=1, keepdims=True)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_sum_jvp_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 3, 4, 5))))
    with deepnet.forward_ad():
        result_tensor = deepnet.sum(a_tensor, dims=(1, 3))
    expected_tangent = np.sum(np.ones((2, 3, 4, 5)), axis=(1, 3))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_sum_jvp_single_element_rank1():
    a = np.random.rand(1)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(1)))
    with deepnet.forward_ad():
        result_tensor = deepnet.sum(a_tensor)
    expected_tangent = np.sum(np.ones(1))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent, rtol=1e-5, atol=1e-5)

def test_squeeze_jvp_rank1_v0():
    a = np.random.rand(1)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(1)))
    with deepnet.forward_ad():
        result_tensor = deepnet.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones(1))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_squeeze_jvp_rank1_v1():
    a = np.random.rand(5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    with deepnet.forward_ad():
        result_tensor = deepnet.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones(5))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_squeeze_jvp_rank2_v0():
    a = np.random.rand(5, 5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((5, 5))))
    with deepnet.forward_ad():
        result_tensor = deepnet.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones((5, 5)))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_squeeze_jvp_rank2_v1():
    a = np.random.rand(3, 1)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 1))))
    with deepnet.forward_ad():
        result_tensor = deepnet.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones((3, 1)))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_squeeze_jvp_multi_v0():
    a = np.random.rand(2, 1, 4, 1, 5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 1, 4, 1, 5))))
    with deepnet.forward_ad():
        result_tensor = deepnet.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones((2, 1, 4, 1, 5)))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_squeeze_jvp_multi_v1():
    a = np.random.rand(1, 1, 1, 1, 1, 50, 1)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((1, 1, 1, 1, 1, 50, 1))))
    with deepnet.forward_ad():
        result_tensor = deepnet.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones((1, 1, 1, 1, 1, 50, 1)))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_squeeze_jvp_multi_v2():
    a = np.random.rand(3, 3, 1, 7)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3, 1, 7))))
    with deepnet.forward_ad():
        result_tensor = deepnet.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones((3, 3, 1, 7)))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_unsqueeze_jvp_multi_v0():
    a = np.random.rand(3, 4, 5)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 4, 5))))
    with deepnet.forward_ad():
        result_tensor = deepnet.unsqueeze(a_tensor, (0, 2))

    expected_tangent = np.expand_dims(np.ones((3, 4, 5)), axis=(0, 2))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_unsqueeze_jvp_multi_v1():
    a = np.random.rand(2, 3)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 3))))
    with deepnet.forward_ad():
        result_tensor = deepnet.unsqueeze(a_tensor, (1, 3, 4))

    expected_tangent = np.expand_dims(np.ones((2, 3)), axis=(1, 3, 4))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_unsqueeze_jvp_multi_v2():
    a = np.random.rand(5, 6, 7, 8)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((5, 6, 7, 8))))
    with deepnet.forward_ad():
        result_tensor = deepnet.unsqueeze(a_tensor, (0, 2, 5))

    expected_tangent = np.expand_dims(np.ones((5, 6, 7, 8)), axis=(0, 2, 5))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_unsqueeze_jvp_multi_v3():
    a = np.random.rand(4, 3)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((4, 3))))
    with deepnet.forward_ad():
        result_tensor = deepnet.unsqueeze(a_tensor, (1,))

    expected_tangent = np.expand_dims(np.ones((4, 3)), axis=1)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_unsqueeze_jvp_multi_v4():
    a = np.random.rand(2, 5, 3)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 5, 3))))
    with deepnet.forward_ad():
        result_tensor = deepnet.unsqueeze(a_tensor, (0, 3))

    expected_tangent = np.expand_dims(np.ones((2, 5, 3)), axis=(0, 3))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_transpose_jvp_multi_v0():
    a = np.random.rand(3, 4, 5)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 4, 5))))
    with deepnet.forward_ad():
        result_tensor = deepnet.transpose(a_tensor, 1, 2)

    expected_tangent = np.swapaxes(np.ones((3, 4, 5)), 1, 2)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_transpose_jvp_multi_v1():
    a = np.random.rand(2, 3)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 3))))
    with deepnet.forward_ad():
        result_tensor = deepnet.transpose(a_tensor, 0, 1)

    expected_tangent = np.swapaxes(np.ones((2, 3)), 0, 1)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_transpose_jvp_multi_v2():
    a = np.random.rand(5, 6, 7, 8)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((5, 6, 7, 8))))
    with deepnet.forward_ad():
        result_tensor = deepnet.transpose(a_tensor, 2, 3)

    expected_tangent = np.swapaxes(np.ones((5, 6, 7, 8)), 2, 3)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_transpose_jvp_multi_v3():
    a = np.random.rand(4, 3)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((4, 3))))
    with deepnet.forward_ad():
        result_tensor = deepnet.transpose(a_tensor, -1, -2)

    expected_tangent = np.swapaxes(np.ones((4, 3)), -1, -2)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_transpose_jvp_multi_v4():
    a = np.random.rand(2, 5, 3)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 5, 3))))
    with deepnet.forward_ad():
        result_tensor = deepnet.transpose(a_tensor, 0, 2)

    expected_tangent = np.swapaxes(np.ones((2, 5, 3)), 0, 2)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_permute_jvp_rank2_v0():
    a = np.random.rand(8, 15)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((8, 15))))
    with deepnet.forward_ad():
        result_tensor = deepnet.permute(a_tensor, (1, 0))

    expected_tangent = np.transpose(np.ones((8, 15)))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_permute_jvp_rank3_v0():
    a = np.random.rand(4, 5, 6)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((4, 5, 6))))
    with deepnet.forward_ad():
        result_tensor = deepnet.permute(a_tensor, (1, 0, 2))

    expected_tangent = np.transpose(np.ones((4, 5, 6)), (1, 0, 2))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_permute_jvp_rank3_v1():
    a = np.random.rand(70, 15, 512)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((70, 15, 512))))
    with deepnet.forward_ad():
        result_tensor = deepnet.permute(a_tensor, (2, 1, 0))

    expected_tangent = np.transpose(np.ones((70, 15, 512)), (2, 1, 0))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_permute_jvp_rank4_v0():
    a = np.random.rand(3, 4, 5, 6)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 4, 5, 6))))
    with deepnet.forward_ad():
        result_tensor = deepnet.permute(a_tensor, (3, 2, 1, 0))

    expected_tangent = np.transpose(np.ones((3, 4, 5, 6)), (3, 2, 1, 0))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_permute_jvp_rank4_v1():
    a = np.random.rand(6, 7, 8, 9)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((6, 7, 8, 9))))
    with deepnet.forward_ad():
        result_tensor = deepnet.permute(a_tensor, (0, 3, 2, 1))

    expected_tangent = np.transpose(np.ones((6, 7, 8, 9)), (0, 3, 2, 1))
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_view_jvp_rank1_to_rank2():
    a = np.random.rand(10)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(10)))
    with deepnet.forward_ad():
        result_tensor = deepnet.view(a_tensor, (2, 5))

    expected_tangent = np.ones(10).reshape(2, 5)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_view_jvp_rank2_to_rank3():
    a = np.random.rand(6, 4)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((6, 4))))
    with deepnet.forward_ad():
        result_tensor = deepnet.view(a_tensor, (2, 3, 4))

    expected_tangent = np.ones((6, 4)).reshape(2, 3, 4)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_view_jvp_rank3_to_rank4():
    a = np.random.rand(2, 3, 4)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 3, 4))))
    with deepnet.forward_ad():
        result_tensor = deepnet.view(a_tensor, (2, 2, 3, 2))

    expected_tangent = np.ones((2, 3, 4)).reshape(2, 2, 3, 2)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_view_jvp_rank2_to_rank1():
    a = np.random.rand(4, 3)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((4, 3))))
    with deepnet.forward_ad():
        result_tensor = deepnet.view(a_tensor, (12,))

    expected_tangent = np.ones((4, 3)).reshape(12)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_view_jvp_rank3_to_rank2():
    a = np.random.rand(2, 3, 4)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 3, 4))))
    with deepnet.forward_ad():
        result_tensor = deepnet.view(a_tensor, (6, 4))

    expected_tangent = np.ones((2, 3, 4)).reshape(6, 4)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_view_jvp_rank4_to_rank2():
    a = np.random.rand(2, 2, 3, 2)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((2, 2, 3, 2))))
    with deepnet.forward_ad():
        result_tensor = deepnet.view(a_tensor, (4, 6))

    expected_tangent = np.ones((2, 2, 3, 2)).reshape(4, 6)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)

def test_view_jvp_with_negative_dim():
    a = np.random.rand(3, 4, 5)
    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 4, 5))))
    with deepnet.forward_ad():
        result_tensor = deepnet.view(a_tensor, (-1, 5))

    expected_tangent = np.ones((3, 4, 5)).reshape(-1, 5)
    np.testing.assert_allclose(result_tensor.tangent.data, expected_tangent)


def main():

    # Add JVP Tests

    test_add_jvp_scalar()
    test_add_jvp_vector()
    test_add_jvp_matrix()

    # Sub JVP Tests

    test_sub_jvp_scalar()
    test_sub_jvp_vector()
    test_sub_jvp_matrix()

    # Mul JVP Tests

    test_mul_jvp_scalar()
    test_mul_jvp_vector()
    test_mul_jvp_matrix()

    # Div JVP Tests

    test_div_jvp_scalar()
    test_div_jvp_vector()
    test_div_jvp_matrix()

    # Matmul JVP Tests

    test_matmul_jvp_square_matrices()
    test_matmul_jvp_different_shapes()
    test_matmul_jvp_higher_rank_same_shape()
    test_matmul_jvp_higher_rank_different_shape()

    # Pow JVP Tests

    test_pow_jvp_scalar()
    test_pow_jvp_vector()
    test_pow_jvp_matrix()

    test_pow_jvp_vector_exp()
    test_pow_jvp_matrix_exp()

    # Exp JVP Tests

    test_exp_jvp_scalar()
    test_exp_jvp_vector()
    test_exp_jvp_matrix()

    # Log JVP Tests

    test_log_jvp_scalar()
    test_log_jvp_vector()
    test_log_jvp_matrix()

    # Sine JVP Tests

    test_sine_jvp_scalar()
    test_sine_jvp_vector()
    test_sine_jvp_matrix()

    # Cosine JVP Tests

    test_cosine_jvp_scalar()
    test_cosine_jvp_vector()
    test_cosine_jvp_matrix()

    # Sum JVP Tests

    test_sum_jvp_single_dim()
    test_sum_jvp_multiple_dim()
    test_sum_jvp_keepdims()
    test_sum_jvp_single_element_rank1()
    test_sum_jvp_higher_rank_tensor()

    # Squeeze JVP Tests

    test_squeeze_jvp_rank1_v0()
    test_squeeze_jvp_rank1_v1()
    test_squeeze_jvp_rank2_v0()
    test_squeeze_jvp_rank2_v1()

    test_squeeze_jvp_multi_v0()
    test_squeeze_jvp_multi_v1()
    test_squeeze_jvp_multi_v2()

    # Unsqueeze JVP Tests

    test_unsqueeze_jvp_multi_v0()
    test_unsqueeze_jvp_multi_v1()
    test_unsqueeze_jvp_multi_v2()
    test_unsqueeze_jvp_multi_v3()
    test_unsqueeze_jvp_multi_v4()

    # Transpose JVP Tests

    test_transpose_jvp_multi_v0()
    test_transpose_jvp_multi_v1()
    test_transpose_jvp_multi_v2()
    test_transpose_jvp_multi_v3()
    test_transpose_jvp_multi_v4()

    # Permute JVP Tests

    test_permute_jvp_rank2_v0()
    test_permute_jvp_rank3_v0()
    test_permute_jvp_rank3_v1()
    test_permute_jvp_rank4_v0()
    test_permute_jvp_rank4_v1()

    # View JVP Tests

    test_view_jvp_rank1_to_rank2()
    test_view_jvp_rank2_to_rank3()
    test_view_jvp_rank3_to_rank4()

    test_view_jvp_rank2_to_rank1()
    test_view_jvp_rank3_to_rank2()
    test_view_jvp_rank4_to_rank2()
    test_view_jvp_with_negative_dim()

    print("All tests passed")

if __name__ == "__main__":
    main()