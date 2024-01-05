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

    print("All tests passed")

if __name__ == "__main__":
    main()