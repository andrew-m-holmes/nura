import numpy as np
import nura
import nura.utils as utils


def test_empty_scalar():
    result = utils.empty(dtype=nura.float).data
    expected = np.empty((), dtype=np.float32)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype


def test_empty_vector():
    result = utils.empty(5, dtype=nura.long).data
    expected = np.empty(5, dtype=np.int64)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype


def test_empty_matrix():
    result = utils.empty(3, 4, dtype=nura.double).data
    expected = np.empty((3, 4), dtype=np.float64)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype


def test_empty_tensor():
    result = utils.empty(2, 3, 4, dtype=nura.int).data
    expected = np.empty((2, 3, 4), dtype=np.int32)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype


def test_empty_float():
    result = utils.empty(3, 3, dtype=nura.float).data
    expected = np.empty((3, 3), dtype=np.float32)
    assert result.dtype == expected.dtype


def test_empty_long():
    result = utils.empty(3, 3, dtype=nura.long).data
    expected = np.empty((3, 3), dtype=np.int64)
    assert result.dtype == expected.dtype


def test_empty_double():
    result = utils.empty(3, 3, dtype=nura.double).data
    expected = np.empty((3, 3), dtype=np.float64)
    assert result.dtype == expected.dtype


def test_empty_int():
    result = utils.empty(3, 3, dtype=nura.int).data
    expected = np.empty((3, 3), dtype=np.int32)
    assert result.dtype == expected.dtype


def test_empty_byte():
    result = utils.empty(3, 3, dtype=nura.byte).data
    expected = np.empty((3, 3), dtype=np.uint8)
    assert result.dtype == expected.dtype


def test_empty_char():
    result = utils.empty(3, 3, dtype=nura.char).data
    expected = np.empty((3, 3), dtype=np.int8)
    assert result.dtype == expected.dtype


def test_empty_half():
    result = utils.empty(3, 3, dtype=nura.half).data
    expected = np.empty((3, 3), dtype=np.float16)
    assert result.dtype == expected.dtype


def test_empty_bool():
    result = utils.empty(3, 3, dtype=nura.bool).data
    expected = np.empty((3, 3), dtype=np.bool_)
    assert result.dtype == expected.dtype


def test_empty_short():
    result = utils.empty(3, 3, dtype=nura.short).data
    expected = np.empty((3, 3), dtype=np.int16)
    assert result.dtype == expected.dtype


def test_zeros_scalar():
    result = utils.zeros(dtype=nura.float).data
    expected = np.zeros((), dtype=np.float32)
    assert np.array_equal(result, expected)


def test_zeros_vector():
    result = utils.zeros(5, dtype=nura.long).data
    expected = np.zeros(5, dtype=np.int64)
    assert np.array_equal(result, expected)


def test_zeros_matrix():
    result = utils.zeros(3, 4, dtype=nura.double).data
    expected = np.zeros((3, 4), dtype=np.float64)
    assert np.array_equal(result, expected)


def test_zeros_tensor():
    result = utils.zeros(2, 3, 4, dtype=nura.int).data
    expected = np.zeros((2, 3, 4), dtype=np.int32)
    assert np.array_equal(result, expected)


def test_zeros_int():
    result = utils.zeros(3, 3, dtype=nura.int).data
    expected = np.zeros((3, 3), dtype=np.int32)
    assert np.array_equal(result, expected)


def test_zeros_long():
    result = utils.zeros(3, 3, dtype=nura.long).data
    expected = np.zeros((3, 3), dtype=np.int64)
    assert np.array_equal(result, expected)


def test_zeros_short():
    result = utils.zeros(3, 3, dtype=nura.short).data
    expected = np.zeros((3, 3), dtype=np.int16)
    assert np.array_equal(result, expected)


def test_zeros_byte():
    result = utils.zeros(3, 3, dtype=nura.byte).data
    expected = np.zeros((3, 3), dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_zeros_char():
    result = utils.zeros(3, 3, dtype=nura.char).data
    expected = np.zeros((3, 3), dtype=np.int8)
    assert np.array_equal(result, expected)


def test_zeros_float():
    result = utils.zeros(3, 3, dtype=nura.float).data
    expected = np.zeros((3, 3), dtype=np.float32)
    assert np.array_equal(result, expected)


def test_zeros_double():
    result = utils.zeros(3, 3, dtype=nura.double).data
    expected = np.zeros((3, 3), dtype=np.float64)
    assert np.array_equal(result, expected)


def test_zeros_half():
    result = utils.zeros(3, 3, dtype=nura.half).data
    expected = np.zeros((3, 3), dtype=np.float16)
    assert np.array_equal(result, expected)


def test_zeros_bool():
    result = utils.zeros(3, 3, dtype=nura.bool).data
    expected = np.zeros((3, 3), dtype=np.bool_)
    assert np.array_equal(result, expected)


def test_ones_scalar():
    result = utils.ones(dtype=nura.float).data
    expected = np.ones((), dtype=np.float32)
    assert np.array_equal(result, expected)


def test_ones_vector():
    result = utils.ones(5, dtype=nura.long).data
    expected = np.ones(5, dtype=np.int64)
    assert np.array_equal(result, expected)


def test_ones_matrix():
    result = utils.ones(3, 4, dtype=nura.double).data
    expected = np.ones((3, 4), dtype=np.float64)
    assert np.array_equal(result, expected)


def test_ones_tensor():
    result = utils.ones(2, 3, 4, dtype=nura.int).data
    expected = np.ones((2, 3, 4), dtype=np.int32)
    assert np.array_equal(result, expected)


def test_ones_int():
    result = utils.ones(3, 3, dtype=nura.int).data
    expected = np.ones((3, 3), dtype=np.int32)
    assert np.array_equal(result, expected)


def test_ones_long():
    result = utils.ones(3, 3, dtype=nura.long).data
    expected = np.ones((3, 3), dtype=np.int64)
    assert np.array_equal(result, expected)


def test_ones_short():
    result = utils.ones(3, 3, dtype=nura.short).data
    expected = np.ones((3, 3), dtype=np.int16)
    assert np.array_equal(result, expected)


def test_ones_byte():
    result = utils.ones(3, 3, dtype=nura.byte).data
    expected = np.ones((3, 3), dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_ones_char():
    result = utils.ones(3, 3, dtype=nura.char).data
    expected = np.ones((3, 3), dtype=np.int8)
    assert np.array_equal(result, expected)


def test_ones_float():
    result = utils.ones(3, 3, dtype=nura.float).data
    expected = np.ones((3, 3), dtype=np.float32)
    assert np.array_equal(result, expected)


def test_ones_double():
    result = utils.ones(3, 3, dtype=nura.double).data
    expected = np.ones((3, 3), dtype=np.float64)
    assert np.array_equal(result, expected)


def test_ones_half():
    result = utils.ones(3, 3, dtype=nura.half).data
    expected = np.ones((3, 3), dtype=np.float16)
    assert np.array_equal(result, expected)


def test_ones_bool():
    result = utils.ones(3, 3, dtype=nura.bool).data
    expected = np.ones((3, 3), dtype=np.bool_)
    assert np.array_equal(result, expected)


def test_randn_scalar():
    result = utils.randn(dtype=nura.float).data
    expected = np.array(np.random.randn())
    assert result.shape == expected.shape


def test_randn_vector():
    result = utils.randn(5, dtype=nura.long).data
    expected = np.random.randn(5)
    assert result.shape == expected.shape


def test_randn_matrix():
    result = utils.randn(3, 4, dtype=nura.double).data
    expected = np.random.randn(3, 4)
    assert result.shape == expected.shape


def test_randn_tensor():
    result = utils.randn(2, 3, 4, dtype=nura.int).data
    expected = np.random.randn(2, 3, 4)
    assert result.shape == expected.shape


def test_randn_int():
    result = utils.randn(3, 3, dtype=nura.int).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.int32
    assert result.shape == expected.shape


def test_randn_long():
    result = utils.randn(3, 3, dtype=nura.long).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.int64
    assert result.shape == expected.shape


def test_randn_short():
    result = utils.randn(3, 3, dtype=nura.short).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.int16
    assert result.shape == expected.shape


def test_randn_byte():
    result = utils.randn(3, 3, dtype=nura.byte).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.uint8
    assert result.shape == expected.shape


def test_randn_char():
    result = utils.randn(3, 3, dtype=nura.char).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.int8
    assert result.shape == expected.shape


def test_randn_float():
    result = utils.randn(3, 3, dtype=nura.float).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.float32
    assert result.shape == expected.shape


def test_randn_double():
    result = utils.randn(3, 3, dtype=nura.double).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.float64
    assert result.shape == expected.shape


def test_randn_half():
    result = utils.randn(3, 3, dtype=nura.half).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.float16
    assert result.shape == expected.shape


def test_randn_bool():
    result = utils.randn(3, 3, dtype=nura.bool).data
    expected = np.random.randn(3, 3)
    assert result.dtype == np.bool_
    assert result.shape == expected.shape


def test_rand_scalar():
    result = utils.rand(dtype=nura.float).data
    assert result.shape == ()


def test_rand_vector():
    result = utils.rand(5, dtype=nura.float).data
    expected_shape = (5,)
    assert result.shape == expected_shape


def test_rand_matrix():
    result = utils.rand(3, 4, dtype=nura.float).data
    expected_shape = (3, 4)
    assert result.shape == expected_shape


def test_rand_tensor():
    result = utils.rand(2, 3, 4, dtype=nura.float).data
    expected_shape = (2, 3, 4)
    assert result.shape == expected_shape


def test_rand_float():
    result = utils.rand(3, 3, dtype=nura.float).data
    assert result.dtype == np.float32


def test_rand_double():
    result = utils.rand(3, 3, dtype=nura.double).data
    assert result.dtype == np.float64


def test_rand_half():
    result = utils.rand(3, 3, dtype=nura.half).data
    assert result.dtype == np.float16


def test_rand_bool():
    result = utils.rand(3, 3, dtype=nura.bool).data
    expected_shape = (3, 3)
    assert result.shape == expected_shape


def test_randint_scalar():
    result = utils.randint(low=0, high=10, dtype=nura.int).data
    assert result.shape == ()


def test_randint_vector():
    result = utils.randint(5, low=0, high=10, dtype=nura.int).data
    assert result.shape == (5,)


def test_randint_matrix():
    result = utils.randint(3, 4, low=0, high=10, dtype=nura.int).data
    assert result.shape == (3, 4)


def test_randint_tensor():
    result = utils.randint(2, 3, 4, low=0, high=10, dtype=nura.int).data
    assert result.shape == (2, 3, 4)


def test_randint_byte():
    result = utils.randint(3, 3, low=0, high=256, dtype=nura.byte).data
    assert result.dtype == np.uint8


def test_randint_char():
    result = utils.randint(3, 3, low=-128, high=128, dtype=nura.char).data
    assert result.dtype == np.int8


def test_randint_short():
    result = utils.randint(3, 3, low=-32768, high=32768, dtype=nura.short).data
    assert result.dtype == np.int16


def test_randint_int():
    result = utils.randint(3, 3, low=-2147483648, high=2147483647, dtype=nura.int).data
    assert result.dtype == np.int32


def test_randint_long():
    result = utils.randint(
        3, 3, low=-9223372036854775808, high=9223372036854775807, dtype=nura.long
    ).data
    assert result.dtype == np.int64


def test_randint_float():
    result = utils.randint(3, 3, low=0, high=10, dtype=nura.float).data
    assert result.dtype == np.float32


def test_randint_double():
    result = utils.randint(3, 3, low=0, high=10, dtype=nura.double).data
    assert result.dtype == np.float64


def test_randint_half():
    result = utils.randint(3, 3, low=0, high=10, dtype=nura.half).data
    assert result.dtype == np.float16


def test_randint_bool():
    result = utils.randint(3, 3, low=0, high=2, dtype=nura.bool).data
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


def test_indexwhere_vector():
    logical = nura.tensor([False, True, False, True], dtype=nura.bool)
    result = nura.indexwhere(logical)
    expected_indices = np.where(logical.data)
    assert all(np.array_equal(r.data, e) for r, e in zip(result, expected_indices))


def test_indexwhere_matrix():
    logical = nura.tensor([[True, False], [False, True]], dtype=nura.bool)
    result = nura.indexwhere(logical)
    expected_indices = np.where(logical.data)
    assert all(np.array_equal(r.data, e) for r, e in zip(result, expected_indices))


def test_indexwhere_tensor():
    logical = nura.tensor(
        [[[True, False], [False, True]], [[False, True], [True, False]]],
        dtype=nura.bool,
    )
    result = nura.indexwhere(logical)
    expected_indices = np.where(logical.data)
    assert all(np.array_equal(r.data, e) for r, e in zip(result, expected_indices))


def test_indexwhere_condition_greater():
    a = nura.tensor([1, 2, 3, 4], dtype=nura.int)
    b = nura.tensor([4, 3, 2, 1], dtype=nura.int)
    logical = a > b
    result = nura.indexwhere(logical)
    expected_indices = np.where(a.data > b.data)
    assert all(np.array_equal(r.data, e) for r, e in zip(result, expected_indices))


def test_indexwhere_condition_less_equal():
    a = nura.tensor([[5, 6], [7, 8]], dtype=nura.int)
    b = nura.tensor([[5, 10], [7, 8]], dtype=nura.int)
    logical = a <= b
    result = nura.indexwhere(logical)
    expected_indices = np.where(a.data <= b.data)
    assert all(np.array_equal(r.data, e) for r, e in zip(result, expected_indices))


def test_indexwhere_condition_equal():
    a = nura.tensor([1, 2, 3, 4, 5], dtype=nura.int)
    b = nura.tensor([5, 1, 2, 3, 4], dtype=nura.int)
    logical = a == b
    result = nura.indexwhere(logical)
    expected_indices = np.where(a.data == b.data)
    assert all(np.array_equal(r.data, e) for r, e in zip(result, expected_indices))


def test_indexwhere_condition_not_equal():
    a = nura.tensor([[1, 2], [3, 4]], dtype=nura.int)
    b = nura.tensor([[2, 1], [4, 3]], dtype=nura.int)
    logical = a != b
    result = nura.indexwhere(logical)
    expected_indices = np.where(a.data != b.data)
    assert all(np.array_equal(r.data, e) for r, e in zip(result, expected_indices))


def test_indexwhere_complex_condition():
    a = nura.tensor([10, 20, 30, 40], dtype=nura.int)
    b = nura.tensor([25, 15, 35, 5], dtype=nura.int)
    logical = (a > b) | (a == 30)
    result = nura.indexwhere(logical)
    expected_indices = np.where((a.data > b.data) | (a.data == 30))
    assert all(np.array_equal(r.data, e) for r, e in zip(result, expected_indices))


def test_to_scalar_byte():
    a = nura.tensor(123, dtype=nura.int)
    result = a.to(nura.byte).data
    expected = np.array(123).astype(np.uint8)
    assert np.array_equal(result, expected) and result.dtype == expected.dtype


def test_to_vector_char():
    a = nura.tensor([1, 2, 3], dtype=nura.int)
    result = a.to(nura.char).data
    expected = np.array([1, 2, 3]).astype(np.int8)
    assert np.array_equal(result, expected) and result.dtype == expected.dtype


def test_to_matrix_short():
    a = nura.tensor([[1, 2], [3, 4]], dtype=nura.int)
    result = a.to(nura.short).data
    expected = np.array([[1, 2], [3, 4]]).astype(np.int16)
    assert np.array_equal(result, expected) and result.dtype == expected.dtype


def test_to_tensor_int():
    a = nura.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=nura.float)
    result = a.to(nura.int).data
    expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    assert np.array_equal(result, expected) and result.dtype == expected.dtype


def test_to_scalar_long():
    a = nura.tensor(123.456, dtype=nura.float)
    result = a.to(nura.long).data
    expected = np.array(123.456).astype(np.int64)
    assert np.array_equal(result, expected) and result.dtype == expected.dtype


def test_to_vector_half():
    a = nura.tensor([1.2, 3.4, 5.6], dtype=nura.float)
    result = a.to(nura.half).data
    expected = np.array([1.2, 3.4, 5.6]).astype(np.float16)
    assert np.array_equal(result, expected) and result.dtype == expected.dtype


def test_to_matrix_float():
    a = nura.tensor([[1, 2], [3, 4]], dtype=nura.int)
    result = a.to(nura.float).data
    expected = np.array([[1, 2], [3, 4]]).astype(np.float32)
    assert np.array_equal(result, expected) and result.dtype == expected.dtype


def test_to_tensor_double():
    a = nura.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=nura.int)
    result = a.to(nura.double).data
    expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float64)
    assert np.array_equal(result, expected) and result.dtype == expected.dtype


def test_to_vector_bool():
    a = nura.tensor([0, 1, 0, 1], dtype=nura.int)
    result = a.to(nura.bool).data
    expected = np.array([0, 1, 0, 1]).astype(np.bool_)
    assert np.array_equal(result, expected) and result.dtype == expected.dtype


def test_onehot_vector_int():
    indices = nura.tensor([0, 2, 1, 3], dtype=nura.int)
    n = 4
    result = nura.onehot(indices, n, dtype=nura.int).data
    expected = np.eye(n, dtype=np.int32)[indices.data]
    assert np.array_equal(result, expected)


def test_onehot_vector_long():
    indices = nura.tensor([1, 0, 3, 2], dtype=nura.long)
    n = 4
    result = nura.onehot(indices, n, dtype=nura.long).data
    expected = np.eye(n, dtype=np.int64)[indices.data]
    assert np.array_equal(result, expected)


def test_onehot_matrix_int():
    indices = nura.tensor([[0, 1], [2, 3]], dtype=nura.int)
    n = 4
    result = nura.onehot(indices, n, dtype=nura.int).data
    expected = np.eye(n, dtype=np.int32)[indices.data]
    assert np.array_equal(result, expected)


def test_onehot_matrix_long():
    indices = nura.tensor([[3, 2], [1, 0]], dtype=nura.long)
    n = 4
    result = nura.onehot(indices, n, dtype=nura.long).data
    expected = np.eye(n, dtype=np.int64)[indices.data]
    assert np.array_equal(result, expected)
