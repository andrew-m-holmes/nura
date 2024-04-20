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


def test_dot_scalar():
    a, b = 2.0, 3.0
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.dot(a_tensor, b_tensor)
    expected = np.dot(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_dot_vector():
    a = np.random.rand(4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.dot(a_tensor, b_tensor)
    expected = np.dot(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_dot_matrix_vector():
    a = np.random.rand(3, 4)
    b = np.random.rand(4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.dot(a_tensor, b_tensor)
    expected = np.dot(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_dot_matrix():
    a = np.random.rand(2, 3)
    b = np.random.rand(3, 4)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.dot(a_tensor, b_tensor)
    expected = np.dot(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_dot_tensor():
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(2, 4, 5)
    a_tensor = nura.tensor(a)
    b_tensor = nura.tensor(b)
    result_tensor = f.dot(a_tensor, b_tensor)
    expected = np.dot(a, b)
    np.testing.assert_allclose(result_tensor.data, expected, rtol=1e-7, atol=1e-7)


def test_dot_different_types():
    a = np.random.rand(3)
    a_tensor = nura.tensor(a)
    result_tensor = nura.dot(a_tensor, 2)
    expected = np.dot(a, 2)
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
