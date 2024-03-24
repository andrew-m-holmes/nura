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
    result = utils.randint(3, 3, low=-9223372036854775808, high=9223372036854775807, dtype=nura.long).data
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
