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

    print("All tests passed")


if __name__ == "__main__":
    main()
