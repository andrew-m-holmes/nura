import nura
import nura.functional as f
import numpy as np


def test_add_scalar():
    a, b = 1.0, 3.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)
    expected = a + b
    np.testing.assert_array_equal(result_tensor.data, expected)


def test_add_vector():
    a = np.random.rand(3)
    b = np.random.rand(3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)
    expected = a + b
    np.testing.assert_array_equal(result_tensor.data, expected)


def test_add_matrix():
    a = np.random.rand(4, 3)
    b = np.random.rand(4, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)
    expected = a + b
    np.testing.assert_array_equal(result_tensor.data, expected)


def test_add_tensor():
    a = np.random.rand(2, 5, 3)
    b = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)
    expected = a + b
    np.testing.assert_array_equal(result_tensor.data, expected)


def test_add_broadcast():
    a = np.random.rand(5, 3, 2)
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)
    expected = a + b
    np.testing.assert_array_equal(result_tensor.data, expected)


def test_sub_scalar():
    a, b = -1.0, 5.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)
    expected = a - b
    np.testing.assert_array_equal(result_tensor.data, expected)


def test_sub_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)
    expected = a - b
    np.testing.assert_array_equal(result_tensor.data, expected)


def test_sub_matrix():
    a = np.random.rand(2, 3)
    b = np.random.rand(2, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)
    expected = a - b
    np.testing.assert_array_equal(result_tensor.data, expected)


def test_sub_tensor():
    a = np.random.rand(4, 2, 5)
    b = np.random.rand(4, 2, 5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)
    expected = a - b
    np.testing.assert_array_equal(result_tensor.data, expected)


def test_sub_broadcast():
    a = np.random.rand(5, 3, 2)
    b = np.random.rand(1, 2)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)
    expected = a - b
    np.testing.assert_array_equal(result_tensor.data, expected)
