import numpy as np
import deepnet
import deepnet.nn.functional as f

# Writing test functions for the 'add' operation using deepnet.tensor


def test_add_forward_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a)
    b_tensor = deepnet.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)

    expected = a + b
    np.testing.assert_almost_equal(result_tensor.data, expected, decimal=5)


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


def main():

    test_add_forward_scalar()


if __name__ == "__main__":
    main()
