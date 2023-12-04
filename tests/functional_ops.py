import numpy as np
import deepnet
import deepnet.nn.functional as f


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

    # Sine Tests

    test_sine_forward_scalar()
    test_sine_forward_vector()
    test_sine_forward_matrix()

    # Cosine Tests

    test_cosine_forward_scalar()
    test_cosine_forward_vector()
    test_cosine_forward_matrix()

    print("All tests passed")


if __name__ == "__main__":
    main()
