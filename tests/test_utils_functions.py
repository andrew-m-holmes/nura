import numpy as np
import nura
import nura.utils as u


def test_empty_scalar():
    result = u.empty(dtype=nura.float).data
    expected = np.empty((), dtype=np.float32)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype


def test_empty_vector():
    result = u.empty(5, dtype=nura.long).data
    expected = np.empty(5, dtype=np.int64)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype


def test_empty_matrix():
    result = u.empty(3, 4, dtype=nura.double).data
    expected = np.empty((3, 4), dtype=np.float64)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype


def test_empty_tensor():
    result = u.empty(2, 3, 4, dtype=nura.int).data
    expected = np.empty((2, 3, 4), dtype=np.int32)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype


def test_empty_float():
    result = u.empty(3, 3, dtype=nura.float).data
    expected = np.empty((3, 3), dtype=np.float32)
    assert result.dtype == expected.dtype


def test_empty_long():
    result = u.empty(3, 3, dtype=nura.long).data
    expected = np.empty((3, 3), dtype=np.int64)
    assert result.dtype == expected.dtype


def test_empty_double():
    result = u.empty(3, 3, dtype=nura.double).data
    expected = np.empty((3, 3), dtype=np.float64)
    assert result.dtype == expected.dtype


def test_empty_int():
    result = u.empty(3, 3, dtype=nura.int).data
    expected = np.empty((3, 3), dtype=np.int32)
    assert result.dtype == expected.dtype


def test_empty_byte():
    result = u.empty(3, 3, dtype=nura.byte).data
    expected = np.empty((3, 3), dtype=np.uint8)
    assert result.dtype == expected.dtype


def test_empty_char():
    result = u.empty(3, 3, dtype=nura.char).data
    expected = np.empty((3, 3), dtype=np.int8)
    assert result.dtype == expected.dtype


def test_empty_half():
    result = u.empty(3, 3, dtype=nura.half).data
    expected = np.empty((3, 3), dtype=np.float16)
    assert result.dtype == expected.dtype


def test_empty_bool():
    result = u.empty(3, 3, dtype=nura.bool).data
    expected = np.empty((3, 3), dtype=np.bool_)
    assert result.dtype == expected.dtype


def test_empty_short():
    result = u.empty(3, 3, dtype=nura.short).data
    expected = np.empty((3, 3), dtype=np.int16)
    assert result.dtype == expected.dtype


def test_zeros_scalar():
    result = u.zeros(dtype=nura.float).data
    expected = np.zeros((), dtype=np.float32)
    assert np.array_equal(result, expected)


def test_zeros_vector():
    result = u.zeros(5, dtype=nura.long).data
    expected = np.zeros(5, dtype=np.int64)
    assert np.array_equal(result, expected)


def test_zeros_matrix():
    result = u.zeros(3, 4, dtype=nura.double).data
    expected = np.zeros((3, 4), dtype=np.float64)
    assert np.array_equal(result, expected)


def test_zeros_tensor():
    result = u.zeros(2, 3, 4, dtype=nura.int).data
    expected = np.zeros((2, 3, 4), dtype=np.int32)
    assert np.array_equal(result, expected)


def test_zeros_int():
    result = u.zeros(3, 3, dtype=nura.int).data
    expected = np.zeros((3, 3), dtype=np.int32)
    assert np.array_equal(result, expected)


def test_zeros_long():
    result = u.zeros(3, 3, dtype=nura.long).data
    expected = np.zeros((3, 3), dtype=np.int64)
    assert np.array_equal(result, expected)


def test_zeros_short():
    result = u.zeros(3, 3, dtype=nura.short).data
    expected = np.zeros((3, 3), dtype=np.int16)
    assert np.array_equal(result, expected)


def test_zeros_byte():
    result = u.zeros(3, 3, dtype=nura.byte).data
    expected = np.zeros((3, 3), dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_zeros_char():
    result = u.zeros(3, 3, dtype=nura.char).data
    expected = np.zeros((3, 3), dtype=np.int8)
    assert np.array_equal(result, expected)


def test_zeros_float():
    result = u.zeros(3, 3, dtype=nura.float).data
    expected = np.zeros((3, 3), dtype=np.float32)
    assert np.array_equal(result, expected)


def test_zeros_double():
    result = u.zeros(3, 3, dtype=nura.double).data
    expected = np.zeros((3, 3), dtype=np.float64)
    assert np.array_equal(result, expected)


def test_zeros_half():
    result = u.zeros(3, 3, dtype=nura.half).data
    expected = np.zeros((3, 3), dtype=np.float16)
    assert np.array_equal(result, expected)


def test_zeros_bool():
    result = u.zeros(3, 3, dtype=nura.bool).data
    expected = np.zeros((3, 3), dtype=np.bool_)
    assert np.array_equal(result, expected)


def test_ones_scalar():
    result = u.ones(dtype=nura.float).data
    expected = np.ones((), dtype=np.float32)
    assert np.array_equal(result, expected)


def test_ones_vector():
    result = u.ones(5, dtype=nura.long).data
    expected = np.ones(5, dtype=np.int64)
    assert np.array_equal(result, expected)


def test_ones_matrix():
    result = u.ones(3, 4, dtype=nura.double).data
    expected = np.ones((3, 4), dtype=np.float64)
    assert np.array_equal(result, expected)


def test_ones_tensor():
    result = u.ones(2, 3, 4, dtype=nura.int).data
    expected = np.ones((2, 3, 4), dtype=np.int32)
    assert np.array_equal(result, expected)


def test_ones_int():
    result = u.ones(3, 3, dtype=nura.int).data
    expected = np.ones((3, 3), dtype=np.int32)
    assert np.array_equal(result, expected)


def test_ones_long():
    result = u.ones(3, 3, dtype=nura.long).data
    expected = np.ones((3, 3), dtype=np.int64)
    assert np.array_equal(result, expected)


def test_ones_short():
    result = u.ones(3, 3, dtype=nura.short).data
    expected = np.ones((3, 3), dtype=np.int16)
    assert np.array_equal(result, expected)


def test_ones_byte():
    result = u.ones(3, 3, dtype=nura.byte).data
    expected = np.ones((3, 3), dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_ones_char():
    result = u.ones(3, 3, dtype=nura.char).data
    expected = np.ones((3, 3), dtype=np.int8)
    assert np.array_equal(result, expected)


def test_ones_float():
    result = u.ones(3, 3, dtype=nura.float).data
    expected = np.ones((3, 3), dtype=np.float32)
    assert np.array_equal(result, expected)


def test_ones_double():
    result = u.ones(3, 3, dtype=nura.double).data
    expected = np.ones((3, 3), dtype=np.float64)
    assert np.array_equal(result, expected)


def test_ones_half():
    result = u.ones(3, 3, dtype=nura.half).data
    expected = np.ones((3, 3), dtype=np.float16)
    assert np.array_equal(result, expected)


def test_ones_bool():
    result = u.ones(3, 3, dtype=nura.bool).data
    expected = np.ones((3, 3), dtype=np.bool_)
    assert np.array_equal(result, expected)


def test_randn_scalar():
    result = u.randn(dtype=nura.float).data
    expected = np.array(np.random.randn())
    assert result.shape == expected.shape


def test_randn_vector():
    result = u.randn(5, dtype=nura.long).data
    expected = np.random.randn(5)
    assert result.shape == expected.shape


def test_randn_matrix():
    result = u.randn(3, 4, dtype=nura.double).data
    expected = np.random.randn(3, 4)
    assert result.shape == expected.shape


def test_randn_tensor():
    result = u.randn(2, 3, 4, dtype=nura.int).data
    expected = np.random.randn(2, 3, 4)
    assert result.shape == expected.shape


def test_randn_int():
    result = u.randn(3, 3, dtype=nura.int).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.int32
    assert result.shape == expected.shape


def test_randn_long():
    result = u.randn(3, 3, dtype=nura.long).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.int64
    assert result.shape == expected.shape


def test_randn_short():
    result = u.randn(3, 3, dtype=nura.short).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.int16
    assert result.shape == expected.shape


def test_randn_byte():
    result = u.randn(3, 3, dtype=nura.byte).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.uint8
    assert result.shape == expected.shape


def test_randn_char():
    result = u.randn(3, 3, dtype=nura.char).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.int8
    assert result.shape == expected.shape


def test_randn_float():
    result = u.randn(3, 3, dtype=nura.float).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.float32
    assert result.shape == expected.shape


def test_randn_double():
    result = u.randn(3, 3, dtype=nura.double).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.float64
    assert result.shape == expected.shape


def test_randn_half():
    result = u.randn(3, 3, dtype=nura.half).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.float16
    assert result.shape == expected.shape


def test_randn_bool():
    result = u.randn(3, 3, dtype=nura.bool).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.bool_
    assert result.shape == expected.shape


def test_rand_scalar():
    result = u.rand(dtype=nura.float).data
    assert result.shape == ()


def test_rand_vector():
    result = u.rand(5, dtype=nura.float).data
    expected_shape = (5,)
    assert result.shape == expected_shape


def test_rand_matrix():
    result = u.rand(3, 4, dtype=nura.float).data
    expected_shape = (3, 4)
    assert result.shape == expected_shape


def test_rand_tensor():
    result = u.rand(2, 3, 4, dtype=nura.float).data
    expected_shape = (2, 3, 4)
    assert result.shape == expected_shape


def test_rand_float():
    result = u.rand(3, 3, dtype=nura.float).data
    assert result.dtype == np.float32


def test_rand_double():
    result = u.rand(3, 3, dtype=nura.double).data
    assert result.dtype == np.float64


def test_rand_half():
    result = u.rand(3, 3, dtype=nura.half).data
    assert result.dtype == np.float16


def test_rand_bool():
    result = u.rand(3, 3, dtype=nura.bool).data
    expected_shape = (3, 3)
    assert result.shape == expected_shape


def test_randint_scalar():
    result = u.randint(low=0, high=10, dtype=nura.int).data
    assert result.shape == ()


def test_randint_vector():
    result = u.randint(5, low=0, high=10, dtype=nura.int).data
    assert result.shape == (5,)


def test_randint_matrix():
    result = u.randint(3, 4, low=0, high=10, dtype=nura.int).data
    assert result.shape == (3, 4)


def test_randint_tensor():
    result = u.randint(2, 3, 4, low=0, high=10, dtype=nura.int).data
    assert result.shape == (2, 3, 4)


def test_randint_byte():
    result = u.randint(3, 3, low=0, high=256, dtype=nura.byte).data
    assert result.dtype == np.uint8


def test_randint_char():
    result = u.randint(3, 3, low=-128, high=128, dtype=nura.char).data
    assert result.dtype == np.int8


def test_randint_short():
    result = u.randint(3, 3, low=-32768, high=32768, dtype=nura.short).data
    assert result.dtype == np.int16


def test_randint_int():
    result = u.randint(3, 3, low=-2147483648, high=2147483647, dtype=nura.int).data
    assert result.dtype == np.int32


def test_randint_long():
    result = u.randint(
        3, 3, low=-9223372036854775808, high=9223372036854775807, dtype=nura.long
    ).data
    assert result.dtype == np.int64


def test_randint_float():
    result = u.randint(3, 3, low=0, high=10, dtype=nura.float).data
    assert result.dtype == np.float32


def test_randint_double():
    result = u.randint(3, 3, low=0, high=10, dtype=nura.double).data
    assert result.dtype == np.float64


def test_randint_half():
    result = u.randint(3, 3, low=0, high=10, dtype=nura.half).data
    assert result.dtype == np.float16


def test_randint_bool():
    result = u.randint(3, 3, low=0, high=2, dtype=nura.bool).data
    assert result.dtype == np.bool_


def test_where_scalar_condition():
    condition = nura.tensor(True, dtype=nura.bool)
    x = nura.tensor(1, dtype=nura.int)
    y = nura.tensor(0, dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(condition.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_vector_condition():
    condition = nura.tensor([True, False, True], dtype=nura.bool)
    x = nura.tensor([1, 2, 3], dtype=nura.int)
    y = nura.tensor([4, 5, 6], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(condition.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_matrix_condition():
    condition = nura.tensor([[True, False], [False, True]], dtype=nura.bool)
    x = nura.tensor([[1, 2], [3, 4]], dtype=nura.int)
    y = nura.tensor([[5, 6], [7, 8]], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(condition.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_tensor_condition():
    condition = nura.tensor(
        [[[True, False], [False, True]], [[True, True], [False, False]]],
        dtype=nura.bool,
    )
    x = nura.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=nura.int)
    y = nura.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(condition.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_condition_tensor_tensor_equal():
    a = nura.tensor([1, 2, 3], dtype=nura.int)
    b = nura.tensor([3, 2, 1], dtype=nura.int)
    condition = a == b
    x = nura.tensor([10, 20, 30], dtype=nura.int)
    y = nura.tensor([40, 50, 60], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(a.data == b.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_condition_tensor_tensor_greater():
    a = nura.tensor([3, 2, 1], dtype=nura.int)
    b = nura.tensor([1, 2, 3], dtype=nura.int)
    condition = a > b
    x = nura.tensor([10, 20, 30], dtype=nura.int)
    y = nura.tensor([40, 50, 60], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(a.data > b.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_condition_tensor_scalar_less():
    a = nura.tensor([1, 2, 3], dtype=nura.int)
    b = 2
    condition = a < b
    x = nura.tensor([10, 20, 30], dtype=nura.int)
    y = nura.tensor([40, 50, 60], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(a.data < b, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_condition_tensor_tensor_not_equal():
    a = nura.tensor([[1, 2], [3, 4]], dtype=nura.int)
    b = nura.tensor([[4, 3], [2, 1]], dtype=nura.int)
    condition = a != b
    x = nura.tensor([[10, 20], [30, 40]], dtype=nura.int)
    y = nura.tensor([[50, 60], [70, 80]], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(a.data != b.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_condition_tensor_tensor_greater_equal():
    a = nura.tensor([3, 4, 5], dtype=nura.int)
    b = nura.tensor([2, 4, 6], dtype=nura.int)
    condition = a >= b
    x = nura.tensor([10, 20, 30], dtype=nura.int)
    y = nura.tensor([40, 50, 60], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(a.data >= b.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_condition_tensor_tensor_less_equal():
    a = nura.tensor([1, 2, 3], dtype=nura.int)
    b = nura.tensor([3, 2, 1], dtype=nura.int)
    condition = a <= b
    x = nura.tensor([10, 20, 30], dtype=nura.int)
    y = nura.tensor([40, 50, 60], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(a.data <= b.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_condition_not():
    condition = ~(nura.tensor([True, False, True], dtype=nura.bool))
    x = nura.tensor([1, 2, 3], dtype=nura.int)
    y = nura.tensor([4, 5, 6], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(~np.array([True, False, True]), x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_condition_and():
    a = nura.tensor([True, False, True], dtype=nura.bool)
    b = nura.tensor([False, False, True], dtype=nura.bool)
    condition = a & b
    x = nura.tensor([10, 20, 30], dtype=nura.int)
    y = nura.tensor([40, 50, 60], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(a.data & b.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_condition_or():
    a = nura.tensor([True, False, False], dtype=nura.bool)
    b = nura.tensor([False, False, True], dtype=nura.bool)
    condition = a | b
    x = nura.tensor([1, 2, 3], dtype=nura.int)
    y = nura.tensor([4, 5, 6], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(a.data | b.data, x.data, y.data)
    assert np.array_equal(result, expected)


def test_where_condition_xor():
    a = nura.tensor([True, False, True, False], dtype=nura.bool)
    b = nura.tensor([True, True, False, False], dtype=nura.bool)
    condition = a ^ b
    x = nura.tensor([1, 2, 3, 4], dtype=nura.int)
    y = nura.tensor([5, 6, 7, 8], dtype=nura.int)
    result = nura.where(condition, x, y).data
    expected = np.where(a.data ^ b.data, x.data, y.data)
    assert np.array_equal(result, expected)
