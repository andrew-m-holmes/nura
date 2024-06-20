import nura
import nura.functional as f
import numpy as np


def test_add_scalar():
    a, b = 1.0, 3.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)
    expected = a + b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_add_vector():
    a = np.random.rand(3)
    b = np.random.rand(3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)
    expected = a + b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_add_matrix():
    a = np.random.rand(4, 3)
    b = np.random.rand(4, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)
    expected = a + b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_add_tensor():
    a = np.random.rand(2, 5, 3)
    b = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)
    expected = a + b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_add_broadcast():
    a = np.random.rand(5, 3, 2)
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.add(a_tensor, b_tensor)
    expected = a + b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_add_operator():
    a = np.random.rand(8, 3)
    b = np.random.rand(2, 8, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = a_tensor + b_tensor
    expected = a + b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_add_inplace():
    a = np.random.rand(4, 2, 3)
    b = np.random.rand(2, 3)
    result_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor += b_tensor
    a += b
    expected = a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_add_different_types():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = nura.add(a_tensor, 3)
    expected = a + 3
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_add_different_types_reversed():
    a = np.random.rand(2)
    a_tensor = nura.tensor(a)
    result_tensor = 4 + a_tensor
    expected = 4 + a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sub_scalar():
    a, b = -1.0, 5.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)
    expected = a - b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sub_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)
    expected = a - b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sub_matrix():
    a = np.random.rand(2, 3)
    b = np.random.rand(2, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)
    expected = a - b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sub_tensor():
    a = np.random.rand(4, 2, 5)
    b = np.random.rand(4, 2, 5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)
    expected = a - b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sub_broadcast():
    a = np.random.rand(5, 3, 2)
    b = np.random.rand(1, 2)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.sub(a_tensor, b_tensor)
    expected = a - b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sub_operator():
    a = np.random.rand(3)
    b = np.random.rand(2, 4, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = a_tensor + b_tensor
    expected = a + b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sub_inplace():
    a = np.random.rand(3)
    b = np.random.rand(1)
    result_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor -= b_tensor
    a -= b
    expected = a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sub_different_types():
    a = np.random.rand(4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = nura.sub(a_tensor, 15.0)
    expected = a - 15.0
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sub_different_types_reversed():
    a = np.random.rand(3, 3, 2)
    a_tensor = nura.tensor(a)
    result_tensor = 0.232 - a_tensor
    expected = 0.232 - a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_mul_scalar():
    a, b = 2.0, 3.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)
    expected = a * b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_mul_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)
    expected = a * b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_mul_matrix():
    a = np.random.rand(3, 4)
    b = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)
    expected = a * b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_mul_tensor():
    a = np.random.rand(2, 4, 3)
    b = np.random.rand(2, 4, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)
    expected = a * b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_mul_broadcast():
    a = np.random.rand(4, 3, 2)
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.mul(a_tensor, b_tensor)
    expected = a * b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_mul_operator():
    a = np.random.rand(6, 3)
    b = np.random.rand(2, 6, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = a_tensor * b_tensor
    expected = a * b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_mul_inplace():
    a = np.random.rand(3, 2, 4)
    b = np.random.rand(2, 4)
    result_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor *= b_tensor
    a *= b
    expected = a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_mul_different_types():
    a = np.random.rand(2, 2, 3)
    a_tensor = nura.tensor(a)
    result_tensor = nura.mul(a_tensor, 2)
    expected = a * 2
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_mul_different_types_reversed():
    a = np.random.rand(3, 2, 1)
    a_tensor = nura.tensor(a)
    result_tensor = 8.2 * a_tensor
    expected = 8.2 * a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_div_scalar():
    a, b = 6.0, 2.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)
    expected = a / b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_div_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)
    expected = a / b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_div_matrix():
    a = np.random.rand(3, 4)
    b = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)
    expected = a / b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_div_tensor():
    a = np.random.rand(2, 4, 3)
    b = np.random.rand(2, 4, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)
    expected = a / b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_div_broadcast():
    a = np.random.rand(4, 3, 2)
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.div(a_tensor, b_tensor)
    expected = a / b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_div_operator():
    a = np.random.rand(6, 3)
    b = np.random.rand(2, 6, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = a_tensor / b_tensor
    expected = a / b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_div_inplace():
    a = np.random.rand(3, 2, 4)
    b = np.random.rand(2, 4)
    result_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor /= b_tensor
    a /= b
    expected = a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_div_different_types():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a)
    result_tensor = nura.div(a_tensor, 2)
    expected = a / 2
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_div_different_types_reversed():
    a = np.random.rand(3)
    a_tensor = nura.tensor(a)
    result_tensor = 6 / a_tensor
    expected = 6 / a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_floordiv_scalar():
    a, b = 7.0, 2.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.floordiv(a_tensor, b_tensor)
    expected = a // b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_floordiv_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.floordiv(a_tensor, b_tensor)
    expected = a // b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_floordiv_matrix():
    a = np.random.rand(2, 5)
    b = np.random.rand(2, 5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.floordiv(a_tensor, b_tensor)
    expected = a // b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_floordiv_tensor():
    a = np.random.rand(3, 2, 4)
    b = np.random.rand(3, 2, 4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.floordiv(a_tensor, b_tensor)
    expected = a // b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_floordiv_broadcast():
    a = np.random.rand(2, 4, 3, 2)
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.floordiv(a_tensor, b_tensor)
    expected = a // b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_floordiv_operator():
    a = np.random.rand(4, 5)
    b = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = a_tensor // b_tensor
    expected = a // b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_floordiv_inplace():
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(3, 4)
    result_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor //= b_tensor
    a //= b
    expected = a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_floordiv_different_types():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a)
    result_tensor = nura.floordiv(a_tensor, 3)
    expected = a // 3
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_floordiv_different_types_reversed():
    a = np.random.rand(3)
    a_tensor = nura.tensor(a)
    result_tensor = 2 // a_tensor
    expected = 2 // a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_modulo_scalar():
    a, b = 7.0, 2.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.modulo(a_tensor, b_tensor)
    expected = a % b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_modulo_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.modulo(a_tensor, b_tensor)
    expected = a % b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_modulo_matrix():
    a = np.random.rand(3, 4)
    b = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.modulo(a_tensor, b_tensor)
    expected = a % b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_modulo_tensor():
    a = np.random.rand(2, 4, 3)
    b = np.random.rand(2, 4, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.modulo(a_tensor, b_tensor)
    expected = a % b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_modulo_broadcast():
    a = np.random.rand(4, 3, 2)
    b = np.random.rand(3, 1)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.modulo(a_tensor, b_tensor)
    expected = a % b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_modulo_operator():
    a = np.random.rand(6, 3)
    b = np.random.rand(2, 6, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = a_tensor % b_tensor
    expected = a % b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_modulo_inplace():
    a = np.random.rand(3, 2, 4)
    b = np.random.rand(2, 4)
    result_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor %= b_tensor
    a %= b
    expected = a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_modulo_different_types():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = nura.modulo(a_tensor, 3)
    expected = a % 3
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_modulo_different_types_reversed():
    a = np.random.rand(2)
    a_tensor = nura.tensor(a)
    result_tensor = 2 % a_tensor
    expected = 2 % a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_dot_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.dot(a_tensor, b_tensor)
    expected = np.dot(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_dot_method():
    a = np.random.rand(4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = a_tensor.dot(b_tensor)
    expected = np.dot(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_matmul_matrix():
    a = np.random.rand(2, 3)
    b = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)
    expected = np.matmul(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_matmul_tensor():
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(2, 4, 5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)
    expected = np.matmul(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_matmul_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)
    b = np.random.rand(2, 3, 5, 6)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)
    expected = np.matmul(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_matmul_vector_tensor():
    a = np.random.rand(4)
    b = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)
    expected = np.matmul(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_matmul_tensor_vector():
    a = np.random.rand(3, 4, 5)
    b = np.random.rand(5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.matmul(a_tensor, b_tensor)
    expected = np.matmul(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_matmul_different_types():
    a = np.random.rand(2, 4, 3)
    a_tensor = nura.tensor(a, dtype=nura.float)
    b = np.random.rand(2, 3, 5)
    b_tensor = nura.tensor(b, dtype=nura.double)
    result_tensor = f.matmul(a_tensor, b_tensor)
    expected = np.matmul(a.astype(np.float32), b.astype(np.float64))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_matmul_operator():
    a = np.random.rand(3, 4)
    b = np.random.rand(4, 5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = a_tensor @ b_tensor
    expected = a @ b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_matmul_inplace():
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(2, 4, 4)
    result_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor @= b_tensor
    a @= b
    expected = a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_pow_scalar():
    a, b = 2.0, 3.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)
    expected = a**b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_pow_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)
    expected = a**b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_pow_matrix():
    a = np.random.rand(2, 5)
    b = np.random.rand(2, 5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)
    expected = a**b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_pow_tensor():
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)
    expected = a**b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_pow_broadcast():
    a = np.random.rand(3, 4, 2)
    b = np.random.rand(4, 1)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.pow(a_tensor, b_tensor)
    expected = a**b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_pow_operator():
    a = np.random.rand(6, 3)
    b = np.random.rand(2, 6, 3)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = a_tensor**b_tensor
    expected = a**b
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_pow_different_types():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = f.pow(a_tensor, 3)
    expected = a**3
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_pow_different_types_reversed():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = 2**a_tensor
    expected = 2**a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_square_scalar():
    a = 2.0
    a_tensor = nura.tensor(a)
    result_tensor = f.square(a_tensor)
    expected = a**2
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_square_vector():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = f.square(a_tensor)
    expected = a**2
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_square_matrix():
    a = np.random.rand(2, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.square(a_tensor)
    expected = a**2
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_square_tensor():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.square(a_tensor)
    expected = a**2
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_square_method():
    a = np.random.rand(6, 3)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.square()
    expected = a**2
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sqrt_scalar():
    a = 4.0
    a_tensor = nura.tensor(a)
    result_tensor = f.sqrt(a_tensor)
    expected = np.sqrt(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sqrt_vector():
    a = np.random.rand(5) + 1e-7
    a_tensor = nura.tensor(a)
    result_tensor = f.sqrt(a_tensor)
    expected = np.sqrt(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sqrt_matrix():
    a = np.random.rand(3, 4) + 1e-7
    a_tensor = nura.tensor(a)
    result_tensor = f.sqrt(a_tensor)
    expected = np.sqrt(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sqrt_tensor():
    a = np.random.rand(2, 4, 3) + 1e-7
    a_tensor = nura.tensor(a)
    result_tensor = f.sqrt(a_tensor)
    expected = np.sqrt(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sqrt_method():
    a = np.random.rand(6, 3) + 1e-7
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.sqrt()
    expected = np.sqrt(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_exp_scalar():
    a = 2.0
    a_tensor = nura.tensor(a).double()
    result_tensor = f.exp(a_tensor)
    expected = np.exp(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_exp_vector():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = f.exp(a_tensor)
    expected = np.exp(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_exp_matrix():
    a = np.random.rand(2, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.exp(a_tensor)
    expected = np.exp(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_exp_tensor():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.exp(a_tensor)
    expected = np.exp(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_exp_method():
    a = np.random.rand(6, 3)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.exp()
    expected = np.exp(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_log_scalar():
    a = 2.0
    a_tensor = nura.tensor(a).double()
    result_tensor = f.log(a_tensor)
    expected = np.log(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_log_vector():
    a = np.random.rand(5) + 1e-7
    a_tensor = nura.tensor(a)
    result_tensor = f.log(a_tensor)
    expected = np.log(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_log_matrix():
    a = np.random.rand(3, 4) + 1e-7
    a_tensor = nura.tensor(a)
    result_tensor = f.log(a_tensor)
    expected = np.log(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_log_tensor():
    a = np.random.rand(2, 4, 3) + 1e-7
    a_tensor = nura.tensor(a)
    result_tensor = f.log(a_tensor)
    expected = np.log(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_log_method():
    a = np.random.rand(6, 3) + 1e-7
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.log()
    expected = np.log(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sin_scalar():
    a = np.random.rand()
    a_tensor = nura.tensor(a).double()
    result_tensor = f.sin(a_tensor)
    expected = np.sin(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sin_vector():
    a = np.random.rand(5)
    a_tensor = nura.tensor(a)
    result_tensor = f.sin(a_tensor)
    expected = np.sin(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sin_matrix():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.sin(a_tensor)
    expected = np.sin(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sin_tensor():
    a = np.random.rand(2, 4, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.sin(a_tensor)
    expected = np.sin(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sin_method():
    a = np.random.rand(6, 3)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.sin()
    expected = np.sin(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_cos_ssincalar():
    a = 5.0
    a_tensor = nura.tensor(a).double()
    result_tensor = f.cos(a_tensor)
    expected = np.cos(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_cos_vector():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = f.cos(a_tensor)
    expected = np.cos(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_cos_matrix():
    a = np.random.rand(2, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.cos(a_tensor)
    expected = np.cos(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_cos_tensor():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.cos(a_tensor)
    expected = np.cos(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_cos_method():
    a = np.random.rand(6, 3)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.cos()
    expected = np.cos(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_scalar():
    a = 7.0
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor)
    expected = np.sum(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_vector():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor)
    expected = np.sum(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_matrix():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor)
    expected = np.sum(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_tensor():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor)
    expected = np.sum(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_method():
    a = np.random.rand(8, 1)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.sum()
    expected = np.sum(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_tuple():
    a = np.random.rand(2, 4, 7)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=(0, 1))
    expected = np.sum(a, axis=(0, 1))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_tuple_keepdims_true():
    a = np.random.rand(1, 3, 2)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=(0, 2), keepdims=True)
    expected = np.sum(a, axis=(0, 2), keepdims=True)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_tuple_keepdims_false():
    a = np.random.rand(4, 2, 1)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=(1, 2), keepdims=False)
    expected = np.sum(a, axis=(1, 2), keepdims=False)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_0_shape_1():
    a = np.random.rand(5, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=0)
    expected = np.sum(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_1_shape_1():
    a = np.random.rand(4, 6)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=1)
    expected = np.sum(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_0_shape_2():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=0)
    expected = np.sum(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_1_shape_2():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=1)
    expected = np.sum(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_2_shape_2():
    a = np.random.rand(4, 2, 6)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=2)
    expected = np.sum(a, axis=2)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_0_shape_3():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=0)
    expected = np.sum(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_1_shape_3():
    a = np.random.rand(3, 4, 2, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=1)
    expected = np.sum(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_2_shape_3():
    a = np.random.rand(4, 3, 5, 2)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=2)
    expected = np.sum(a, axis=2)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_sum_dim_3_shape_3():
    a = np.random.rand(2, 4, 3, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.sum(a_tensor, dim=3)
    expected = np.sum(a, axis=3)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_scalar():
    a = 2.0
    a_tensor = nura.tensor(a).double()
    result_tensor = f.max(a_tensor)
    expected = np.max(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_vector():
    a = np.random.rand(7)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor)
    expected = np.max(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_matrix():
    a = np.random.rand(4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor)
    expected = np.max(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_tensor():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor)
    expected = np.max(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_method():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.max()
    expected = np.max(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_tuple():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=(0, 1))
    expected = np.max(a, axis=(0, 1))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_tuple_keepdims_true():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=(0, 2), keepdims=True)
    expected = np.max(a, axis=(0, 2), keepdims=True)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_tuple_keepdims_false():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=(1, 2), keepdims=False)
    expected = np.max(a, axis=(1, 2), keepdims=False)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_0_shape_1():
    a = np.random.rand(5, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=0)
    expected = np.max(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_1_shape_1():
    a = np.random.rand(4, 6)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=1)
    expected = np.max(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_0_shape_2():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=0)
    expected = np.max(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_1_shape_2():
    a = np.random.rand(2, 5, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=1)
    expected = np.max(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_2_shape_2():
    a = np.random.rand(4, 2, 6)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=2)
    expected = np.max(a, axis=2)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_0_shape_3():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=0)
    expected = np.max(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_1_shape_3():
    a = np.random.rand(3, 4, 2, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=1)
    expected = np.max(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_2_shape_3():
    a = np.random.rand(4, 3, 5, 2)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=2)
    expected = np.max(a, axis=2)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_max_dim_3_shape_3():
    a = np.random.rand(2, 4, 3, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.max(a_tensor, dim=3)
    expected = np.max(a, axis=3)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_scalar():
    a = 3.0
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor)
    expected = np.min(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_vector():
    a = np.random.rand(6)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor)
    expected = np.min(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_matrix():
    a = np.random.rand(3, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor)
    expected = np.min(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_tensor():
    a = np.random.rand(2, 4, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor)
    expected = np.min(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_method():
    a = np.random.rand(3, 2, 5)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.min()
    expected = np.min(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_tuple():
    a = np.random.rand(2, 4, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=(0, 2))
    expected = np.min(a, axis=(0, 2))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_tuple_keepdims_true():
    a = np.random.rand(3, 2, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=(1, 2), keepdims=True)
    expected = np.min(a, axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_tuple_keepdims_false():
    a = np.random.rand(4, 2, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=(0, 1), keepdims=False)
    expected = np.min(a, axis=(0, 1), keepdims=False)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_0_shape_1():
    a = np.random.rand(6, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=0)
    expected = np.min(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_1_shape_1():
    a = np.random.rand(3, 7)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=1)
    expected = np.min(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_0_shape_2():
    a = np.random.rand(4, 3, 2)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=0)
    expected = np.min(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_1_shape_2():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=1)
    expected = np.min(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_2_shape_2():
    a = np.random.rand(5, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=2)
    expected = np.min(a, axis=2)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_0_shape_3():
    a = np.random.rand(3, 2, 5, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=0)
    expected = np.min(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_1_shape_3():
    a = np.random.rand(2, 5, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=1)
    expected = np.min(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_2_shape_3():
    a = np.random.rand(5, 4, 2, 3)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=2)
    expected = np.min(a, axis=2)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_min_dim_3_shape_3():
    a = np.random.rand(3, 5, 4, 2)
    a_tensor = nura.tensor(a)
    result_tensor = f.min(a_tensor, dim=3)
    expected = np.min(a, axis=3)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)

# TODO add tests for mean, var, and std

def test_transpose_matrix():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.transpose(a_tensor)
    expected = np.swapaxes(a, -2, -1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_transpose_tensor():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.transpose(a_tensor)
    expected = np.swapaxes(a, -2, -1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_transpose_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.transpose(a_tensor, 0, 2)
    expected = np.swapaxes(a, 0, 2)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_transpose_method():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.transpose()
    expected = np.swapaxes(a, -2, -1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_transpose_custom_dims():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.transpose(a_tensor, dim0=1, dim1=3)
    expected = np.swapaxes(a, 1, 3)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_transpose_negative_dims():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.transpose(a_tensor, dim0=-3, dim1=-1)
    expected = np.swapaxes(a, -3, -1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_transpose_same_dims():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.transpose(a_tensor, dim0=2, dim1=2)
    expected = a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_permute_vector():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = f.permute(a_tensor, dims=(0,))
    expected = a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_permute_matrix():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.permute(a_tensor, dims=(1, 0))
    expected = np.transpose(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_permute_tensor():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.permute(a_tensor, dims=(2, 0, 1))
    expected = np.transpose(a, axes=(2, 0, 1))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_permute_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.permute(a_tensor, dims=(3, 1, 2, 0))
    expected = np.transpose(a, axes=(3, 1, 2, 0))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_permute_method():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.permute(dims=(2, 0, 1))
    expected = np.transpose(a, axes=(2, 0, 1))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_permute_inverse_permutation():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    permuted_tensor = f.permute(a_tensor, dims=(3, 1, 2, 0))
    result_tensor = f.permute(permuted_tensor, dims=(3, 1, 2, 0))
    expected = a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_squeeze_scalar():
    a = np.array(3.14)
    a_tensor = nura.tensor(a)
    result_tensor = f.squeeze(a_tensor)
    expected = np.squeeze(a)
    assert isinstance(result_tensor.data, np.ndarray)
    assert result_tensor.data.shape == ()
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_squeeze_vector():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = f.squeeze(a_tensor)
    expected = np.squeeze(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_squeeze_matrix():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.squeeze(a_tensor)
    expected = np.squeeze(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_squeeze_tensor():
    a = np.random.rand(1, 2, 3, 1, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.squeeze(a_tensor)
    expected = np.squeeze(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_squeeze_method():
    a = np.random.rand(1, 3, 1, 4, 1)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.squeeze()
    expected = np.squeeze(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_squeeze_specific_dimension():
    a = np.random.rand(1, 2, 3, 1, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.squeeze(a_tensor, dim=3)
    expected = np.squeeze(a, axis=3)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_squeeze_multiple_dimensions():
    a = np.random.rand(1, 2, 1, 3, 1, 4, 1)
    a_tensor = nura.tensor(a)
    result_tensor = f.squeeze(a_tensor, dim=(0, 2, 4, 6))
    expected = np.squeeze(a, axis=(0, 2, 4, 6))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_squeeze_no_singleton_dimensions():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.squeeze(a_tensor)
    expected = np.squeeze(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_unsqueeze_scalar():
    a = np.array(3.14)
    a_tensor = nura.tensor(a)
    result_tensor = f.unsqueeze(a_tensor, dim=0)
    expected = np.expand_dims(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_unsqueeze_vector():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = f.unsqueeze(a_tensor, dim=1)
    expected = np.expand_dims(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_unsqueeze_matrix():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.unsqueeze(a_tensor, dim=0)
    expected = np.expand_dims(a, axis=0)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_unsqueeze_tensor():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.unsqueeze(a_tensor, dim=2)
    expected = np.expand_dims(a, axis=2)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_unsqueeze_high_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.unsqueeze(a_tensor, dim=1)
    expected = np.expand_dims(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_unsqueeze_method():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.unsqueeze(dim=-1)
    expected = np.expand_dims(a, axis=-1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_unsqueeze_multiple_dimensions():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.unsqueeze(a_tensor, dim=(0, 1))
    expected = np.expand_dims(np.expand_dims(a, axis=0), axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_unsqueeze_negative_dimension():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.unsqueeze(a_tensor, dim=-2)
    expected = np.expand_dims(a, axis=-2)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_unsqueeze_in_place_dimension():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.unsqueeze(a_tensor, dim=1)
    expected = np.expand_dims(a, axis=1)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_reshape_scalar():
    a = np.array(3.14)
    a_tensor = nura.tensor(a)
    result_tensor = f.reshape(a_tensor, newdim=())
    expected = np.reshape(a, newshape=())
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_reshape_vector():
    a = np.random.rand(4)
    a_tensor = nura.tensor(a)
    result_tensor = f.reshape(a_tensor, newdim=(2, 2))
    expected = np.reshape(a, newshape=(2, 2))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_reshape_matrix():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.reshape(a_tensor, newdim=(6, 2))
    expected = np.reshape(a, newshape=(6, 2))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_reshape_tensor():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.reshape(a_tensor, newdim=(4, 6))
    expected = np.reshape(a, newshape=(4, 6))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_reshape_method():
    a = np.random.rand(3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.reshape(newdim=(12, 5))
    expected = np.reshape(a, newshape=(12, 5))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_reshape_higher_dimensions():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.reshape(a_tensor, newdim=(6, 4, 5))
    expected = np.reshape(a, newshape=(6, 4, 5))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_reshape_unknown_dimension():
    a = np.random.rand(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.reshape(a_tensor, newdim=(4, -1))
    expected = np.reshape(a, newshape=(4, -1))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_reshape_to_scalar():
    a = np.random.rand(1, 1, 1)
    a_tensor = nura.tensor(a)
    result_tensor = f.reshape(a_tensor, newdim=())
    expected = np.reshape(a, newshape=())
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_reshape_to_vector():
    a = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.reshape(a_tensor, newdim=(12,))
    expected = np.reshape(a, newshape=(12,))
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_abs_scalar():
    a = np.array(-3.14)
    a_tensor = nura.tensor(a)
    result_tensor = f.abs(a_tensor)
    expected = np.abs(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_abs_vector():
    a = np.random.randn(5)
    a_tensor = nura.tensor(a)
    result_tensor = f.abs(a_tensor)
    expected = np.abs(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_abs_matrix():
    a = np.random.randn(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.abs(a_tensor)
    expected = np.abs(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_abs_tensor():
    a = np.random.randn(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.abs(a_tensor)
    expected = np.abs(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_abs_method():
    a = np.random.randn(3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.abs()
    expected = np.abs(a)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_pos_scalar():
    a = np.array(-3.14)
    a_tensor = nura.tensor(a)
    result_tensor = f.pos(a_tensor)
    expected = +a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_pos_vector():
    a = np.random.randn(5)
    a_tensor = nura.tensor(a)
    result_tensor = f.pos(a_tensor)
    expected = +a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_pos_matrix():
    a = np.random.randn(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.pos(a_tensor)
    expected = +a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_pos_tensor():
    a = np.random.randn(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.pos(a_tensor)
    expected = +a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_pos_operator():
    a = np.random.randn(3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = +a_tensor
    expected = +a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_neg_scalar():
    a = np.array(2.5)
    a_tensor = nura.tensor(a)
    result_tensor = f.neg(a_tensor)
    expected = -a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_neg_vector():
    a = np.random.randn(4)
    a_tensor = nura.tensor(a)
    result_tensor = f.neg(a_tensor)
    expected = -a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_neg_matrix():
    a = np.random.randn(2, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.neg(a_tensor)
    expected = -a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_neg_tensor():
    a = np.random.randn(3, 2, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.neg(a_tensor)
    expected = -a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_neg_operator():
    a = np.random.randn(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = -a_tensor
    expected = -a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_neg_scalar_zero():
    a = np.array(0.0)
    a_tensor = nura.tensor(a)
    result_tensor = f.neg(a_tensor)
    expected = -a
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-8, atol=1e-8)


def test_clone_scalar():
    a = np.array(3.7)
    a_tensor = nura.tensor(a)
    result_tensor = f.clone(a_tensor)
    assert result_tensor is not a_tensor
    assert result_tensor.data is not a_tensor.data
    np.testing.assert_allclose(result_tensor.data, a_tensor.data, rtol=1e-8, atol=1e-8)


def test_clone_vector():
    a = np.random.randn(5)
    a_tensor = nura.tensor(a)
    result_tensor = f.clone(a_tensor)
    assert result_tensor is not a_tensor
    assert result_tensor.data is not a_tensor.data
    np.testing.assert_allclose(result_tensor.data, a_tensor.data, rtol=1e-8, atol=1e-8)


def test_clone_matrix():
    a = np.random.randn(3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.clone(a_tensor)
    assert result_tensor is not a_tensor
    assert result_tensor.data is not a_tensor.data
    np.testing.assert_allclose(result_tensor.data, a_tensor.data, rtol=1e-8, atol=1e-8)


def test_clone_tensor():
    a = np.random.randn(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.clone(a_tensor)
    assert result_tensor is not a_tensor
    assert result_tensor.data is not a_tensor.data
    np.testing.assert_allclose(result_tensor.data, a_tensor.data, rtol=1e-8, atol=1e-8)


def test_clone_method():
    a = np.random.randn(2, 3, 4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor.clone()
    assert result_tensor is not a_tensor
    assert result_tensor.data is not a_tensor.data
    np.testing.assert_allclose(result_tensor.data, a_tensor.data, rtol=1e-8, atol=1e-8)


def test_select_scalar():
    a = np.random.randn(3, 7)
    a_tensor = nura.tensor(a)
    result_tensor = f.select(a_tensor, (0, 1))
    expected = a[0, 1]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_vector():
    a = np.random.randn(7)
    a_tensor = nura.tensor(a)
    result_tensor = f.select(a_tensor, (3,))
    expected = a[3]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_matrix():
    a = np.random.randn(4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = f.select(a_tensor, (2, 3))
    expected = a[2, 3]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_tensor():
    a = np.random.randn(2, 3, 4)
    a_tensor = nura.tensor(a)
    result_tensor = f.select(a_tensor, (1, 2, 3))
    expected = a[1, 2, 3]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_slice():
    a = np.random.randn(4, 5, 6)
    a_tensor = nura.tensor(a)
    result_tensor = f.select(a_tensor, (slice(1, 3), slice(2, 5), slice(1, 4)))
    expected = a[1:3, 2:5, 1:4]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_combined():
    a = np.random.randn(5, 6, 7)
    a_tensor = nura.tensor(a)
    result_tensor = f.select(a_tensor, (2, slice(1, 4), 3))
    expected = a[2, 1:4, 3]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_operator():
    a = np.random.randn(4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor[1, 3]
    expected = a[1, 3]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_slice_all_rows():
    a = np.random.randn(4, 5)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor[:, 2]
    expected = a[:, 2]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_ellipsis():
    a = np.random.randn(4, 5, 6)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor[..., 3]
    expected = a[..., 3]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_ellipsis_middle():
    a = np.random.randn(4, 5, 6)
    a_tensor = nura.tensor(a)
    result_tensor = a_tensor[:, ..., 4]
    expected = a[:, ..., 4]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_with_tensor_index():
    a = np.random.randn(4, 5)
    indices = np.array([0, 2])
    a_tensor = nura.tensor(a)
    indices_tensor = nura.tensor(indices)
    result_tensor = a_tensor[indices_tensor]
    expected = a[indices]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_with_tensor_index_2d():
    a = np.random.randn(4, 5, 6)
    indices = np.array([[0, 1], [2, 3]])
    a_tensor = nura.tensor(a)
    indices_tensor = nura.tensor(indices)
    result_tensor = a_tensor[indices_tensor, 1, 2]
    expected = a[indices, 1, 2]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_select_with_tensor_index_3d():
    a = np.random.randn(4, 5, 6)
    indices = np.array([[[0, 1], [2, 3]], [[1, 0], [3, 2]]])
    a_tensor = nura.tensor(a)
    indices_tensor = nura.tensor(indices)
    result_tensor = a_tensor[indices_tensor, 2, 1]
    expected = a[indices, 2, 1]
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)
