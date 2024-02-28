import numpy as np
import neuro
import neuro.functional as f


def test_add_tangent_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(1.0))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.add(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h + b - (a - h + b)) / (2 * h)
    expected_b = (a + b + h - (a + b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_add_tangent_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.add(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h + b - (a - h + b)) / (2 * h)
    expected_b = (a + b + h - (a + b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_add_tangent_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.add(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h + b - (a - h + b)) / (2 * h)
    expected_b = (a + b + h - (a + b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_sub_tangent_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(1.0))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.sub(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h - b - (a - h - b)) / (2 * h)
    expected_b = (a - (b + h) - (a - (b - h))) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_sub_tangent_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.sub(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h - b - (a - h - b)) / (2 * h)
    expected_b = (a - (b + h) - (a - (b - h))) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_sub_tangent_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.sub(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h - b - (a - h - b)) / (2 * h)
    expected_b = (a - (b + h) - (a - (b - h))) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_mul_tangent_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(1.0))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.mul(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) * b - (a - h) * b) / (2 * h)
    expected_b = (a * (b + h) - a * (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_mul_tangent_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.mul(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) * b - (a - h) * b) / (2 * h)
    expected_b = (a * (b + h) - a * (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_mul_tangent_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.mul(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) * b - (a - h) * b) / (2 * h)
    expected_b = (a * (b + h) - a * (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_div_tangent_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(1.0))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.div(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_b = (a / (b + h) - a / (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_div_tangent_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.div(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_b = (a / (b + h) - a / (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_div_tangent_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.div(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_b = (a / (b + h) - a / (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(result_tensor.grad.data, expected, rtol=1e-5, atol=1e-5)


def test_dot_tangent_vector_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.dot(a_tensor, b_tensor)

    expected_tangent = np.sum(a) + np.sum(b)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_dot_tangent_matrix_vector():
    a = np.random.rand(4, 3)
    b = np.random.rand(3)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(np.ones(3)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.dot(a_tensor, b_tensor)

    expected_tangent = np.dot(np.ones((4, 3)), b) + np.dot(a, np.ones(3))
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_dot_tangent_vector_matrix():
    a = np.random.rand(3)
    b = np.random.rand(3, 5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(3)))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.dot(a_tensor, b_tensor)

    expected_tangent = np.dot(np.ones(3), b) + np.dot(a, np.ones((3, 5)))
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_dot_tangent_matrix_matrix():
    a = np.random.rand(4, 3)
    b = np.random.rand(3, 5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.dot(a_tensor, b_tensor)

    expected_tangent = np.dot(a, np.ones((3, 5))) + np.dot(np.ones((4, 3)), b)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_matmul_tangent_square_matrices():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((5, 5)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((5, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.matmul(a_tensor, b_tensor)

    expected_tangent = np.matmul(a, np.ones((5, 5))) + np.matmul(np.ones((5, 5)), b)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_matmul_tangent_different_shapes():
    a = np.random.rand(4, 3)
    b = np.random.rand(3, 5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.matmul(a_tensor, b_tensor)

    expected_tangent = np.matmul(a, np.ones((3, 5))) + np.matmul(np.ones((4, 3)), b)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_matmul_tangent_higher_rank_same_shape():
    a = np.random.rand(2, 3, 4, 4)
    b = np.random.rand(2, 3, 4, 4)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3, 4, 4)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3, 4, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.matmul(a_tensor, b_tensor)

    expected_tangent = np.matmul(a, np.ones((2, 3, 4, 4))) + np.matmul(
        np.ones((2, 3, 4, 4)), b
    )
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_matmul_tangent_higher_rank_different_shape():
    a = np.random.rand(2, 4, 3)
    b = np.random.rand(2, 3, 5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 4, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.matmul(a_tensor, b_tensor)

    expected_tangent = np.matmul(a, np.ones((2, 3, 5))) + np.matmul(
        np.ones((2, 4, 3)), b
    )
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_matmul_tangent_different_ranks():
    a = np.random.rand(2, 4, 3)
    b = np.random.rand(3, 2)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 4, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 2)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.matmul(a_tensor, b_tensor)

    expected_tangent = np.matmul(a, np.ones((3, 2))) + np.matmul(np.ones((2, 4, 3)), b)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_pow_tangent_scalar():
    a = np.random.rand()
    b = 2.0

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(1.0))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.pow(a_tensor, b_tensor)

    h = 1e-8
    expected_tangent_a = ((a + h) ** b - (a - h) ** b) / (2 * h)
    expected_tangent_b = (a ** (b + h) - a ** (b - h)) / (2 * h)
    expected_tangent = expected_tangent_a + expected_tangent_b
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_pow_tangent_vector():
    a = np.random.rand(4)
    b = 3.0

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(4)))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.pow(a_tensor, b_tensor)

    h = 1e-8
    expected_tangent_a = ((a + h) ** b - (a - h) ** b) / (2 * h)
    expected_tangent_b = (a ** (b + h) - a ** (b - h)) / (2 * h)
    expected_tangent = expected_tangent_a + expected_tangent_b
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_pow_tangent_matrix():
    a = np.random.rand(3, 3)
    b = 4.0

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.pow(a_tensor, b_tensor)

    h = 1e-8
    expected_tangent_a = ((a + h) ** b - (a - h) ** b) / (2 * h)
    expected_tangent_b = (a ** (b + h) - a ** (b - h)) / (2 * h)
    expected_tangent = expected_tangent_a + expected_tangent_b
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_pow_tangent_vector_exp():
    a = np.random.rand(4)
    b = np.full_like(a, 2)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(4)))
    b_tensor = neuro.tensor(b, usegrad=True).mutated(grad=neuro.tensor(np.ones(4)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.pow(a_tensor, b_tensor)

    h = 1e-8
    expected_tangent_a = ((a + h) ** b - (a - h) ** b) / (2 * h)
    expected_tangent_b = (a ** (b + h) - a ** (b - h)) / (2 * h)
    expected_tangent = expected_tangent_a + expected_tangent_b
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_pow_tangent_matrix_exp():
    a = np.random.rand(3, 3)
    b = np.full_like(a, 3)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    b_tensor = neuro.tensor(b, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.pow(a_tensor, b_tensor)

    h = 1e-8
    expected_tangent_a = ((a + h) ** b - (a - h) ** b) / (2 * h)
    expected_tangent_b = (a ** (b + h) - a ** (b - h)) / (2 * h)
    expected_tangent = expected_tangent_a + expected_tangent_b
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_exp_tangent_scalar():
    a = np.random.rand()

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.exp(a_tensor)

    h = 1e-8
    expected_tangent = (np.exp(a + h) - np.exp(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_exp_tangent_vector():
    a = np.random.rand(5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.exp(a_tensor)

    h = 1e-8
    expected_tangent = (np.exp(a + h) - np.exp(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_exp_tangent_matrix():
    a = np.random.rand(3, 4)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.exp(a_tensor)

    h = 1e-8
    expected_tangent = (np.exp(a + h) - np.exp(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_log_tangent_scalar():
    a = np.random.rand()

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.log(a_tensor)

    h = 1e-8
    expected_tangent = (np.log(a + h) - np.log(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_log_tangent_vector():
    a = np.random.rand(5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.log(a_tensor)

    h = 1e-8
    expected_tangent = (np.log(a + h) - np.log(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_log_tangent_matrix():
    a = np.random.rand(3, 4)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.log(a_tensor)

    h = 1e-8
    expected_tangent = (np.log(a + h) - np.log(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_sin_tangent_scalar():
    a = np.random.rand()
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.sin(a_tensor)
    h = 1e-8
    expected_tangent = (np.sin(a + h) - np.sin(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_sin_tangent_vector():
    a = np.random.rand(4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(4)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.sin(a_tensor)
    h = 1e-8
    expected_tangent = (np.sin(a + h) - np.sin(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_sin_tangent_matrix():
    a = np.random.rand(3, 3)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.sin(a_tensor)
    h = 1e-8
    expected_tangent = (np.sin(a + h) - np.sin(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_cos_tangent_scalar():
    a = np.random.rand()
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.cos(a_tensor)
    h = 1e-8
    expected_tangent = (np.cos(a + h) - np.cos(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_cos_tangent_vector():
    a = np.random.rand(4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(4)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.cos(a_tensor)
    h = 1e-8
    expected_tangent = (np.cos(a + h) - np.cos(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_cos_tangent_matrix():
    a = np.random.rand(4, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.cos(a_tensor)
    h = 1e-8
    expected_tangent = (np.cos(a + h) - np.cos(a - h)) / (2 * h)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_sum_tangent_single_dim():
    a = np.random.rand(3, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.sum(a_tensor, dim=1)
    expected_tangent = np.sum(np.ones((3, 4)), axis=1)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_sum_tangent_multiple_dim():
    a = np.random.rand(3, 4, 5)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.sum(a_tensor, dim=(1, 2))
    expected_tangent = np.sum(np.ones((3, 4, 5)), axis=(1, 2))
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_sum_tangent_keepdims():
    a = np.random.rand(3, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.sum(a_tensor, dim=1, keepdims=True)
    expected_tangent = np.sum(np.ones((3, 4)), axis=1, keepdims=True)
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_sum_tangent_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3, 4, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.sum(a_tensor, dim=(1, 3))
    expected_tangent = np.sum(np.ones((2, 3, 4, 5)), axis=(1, 3))
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_sum_tangent_single_element_rank1():
    a = np.random.rand(1)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(1)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.sum(a_tensor)
    expected_tangent = np.sum(np.ones(1))
    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_max_tangent_single_dim():
    a = np.random.rand(3, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.max(a_tensor, dim=1)

    mask = a == result_tensor.data.reshape(-1, 1)
    pretangent = np.where(mask, a_tensor.grad.data, -np.inf)
    expected_tangent = np.max(pretangent, axis=1)

    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_max_tangent_multiple_dim():
    a = np.random.rand(3, 4, 5)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.max(a_tensor, dim=(1, 2))

    mask = a == result_tensor.data[..., None, None]
    pretangent = np.where(mask, a_tensor.grad.data, -np.inf)
    expected_tangent = np.max(pretangent, axis=(1, 2))

    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_max_tangent_keepdims():
    a = np.random.rand(3, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.max(a_tensor, dim=1, keepdims=True)

    mask = a == result_tensor.data
    pretangent = np.where(mask, a_tensor.grad.data, -np.inf)
    expected_tangent = np.max(pretangent, axis=1, keepdims=True)

    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_max_tangent_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3, 4, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.max(a_tensor, dim=(1, 3))

    mask = a == result_tensor.data.reshape(2, 1, 4, 1)
    pretangent = np.where(mask, a_tensor.grad.data, -np.inf)
    expected_tangent = np.max(pretangent, axis=(1, 3))

    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_max_tangent_single_element_rank1():
    a = np.random.rand(1)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(1)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.max(a_tensor)

    mask = a == result_tensor.data
    pretangent = np.where(mask, a_tensor.grad.data, -np.inf)
    expected_tangent = np.max(pretangent)

    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_min_tangent_single_dim():
    a = np.random.rand(3, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.min(a_tensor, dim=1)

    mask = a == result_tensor.data.reshape(-1, 1)
    pretangent = np.where(mask, a_tensor.grad.data, np.inf)
    expected_tangent = np.min(pretangent, axis=1)

    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_min_tangent_multiple_dim():
    a = np.random.rand(3, 4, 5)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.min(a_tensor, dim=(1, 2))

    mask = a == result_tensor.data[..., None, None]
    pretangent = np.where(mask, a_tensor.grad.data, np.inf)
    expected_tangent = np.min(pretangent, axis=(1, 2))

    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_min_tangent_keepdims():
    a = np.random.rand(3, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.min(a_tensor, dim=1, keepdims=True)

    mask = a == result_tensor.data
    pretangent = np.where(mask, a_tensor.grad.data, np.inf)
    expected_tangent = np.min(pretangent, axis=1, keepdims=True)

    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_min_tangent_higher_rank_tensor():
    a = np.random.rand(2, 3, 4, 5)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3, 4, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.min(a_tensor, dim=(1, 3))

    mask = a == result_tensor.data.reshape(2, 1, 4, 1)
    pretangent = np.where(mask, a_tensor.grad.data, np.inf)
    expected_tangent = np.min(pretangent, axis=(1, 3))

    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_min_tangent_single_element_rank1():
    a = np.random.rand(1)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(1)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = f.min(a_tensor)

    mask = a == result_tensor.data
    pretangent = np.where(mask, a_tensor.grad.data, np.inf)
    expected_tangent = np.min(pretangent)

    np.testing.assert_allclose(
        result_tensor.grad.data, expected_tangent, rtol=1e-5, atol=1e-5
    )


def test_squeeze_tangent_rank1_v0():
    a = np.random.rand(1)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(1)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones(1))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_squeeze_tangent_rank1_v1():
    a = np.random.rand(5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones(5))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_squeeze_tangent_rank2_v0():
    a = np.random.rand(5, 5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((5, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones((5, 5)))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_squeeze_tangent_rank2_v1():
    a = np.random.rand(3, 1)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 1)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones((3, 1)))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_squeeze_tangent_multi_v0():
    a = np.random.rand(2, 1, 4, 1, 5)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 1, 4, 1, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones((2, 1, 4, 1, 5)))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_squeeze_tangent_multi_v1():
    a = np.random.rand(1, 1, 1, 1, 1, 50, 1)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((1, 1, 1, 1, 1, 50, 1)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones((1, 1, 1, 1, 1, 50, 1)))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_squeeze_tangent_multi_v2():
    a = np.random.rand(3, 3, 1, 7)

    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 3, 1, 7)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.squeeze(a_tensor)

    expected_tangent = np.squeeze(np.ones((3, 3, 1, 7)))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_unsqueeze_tangent_multi_v0():
    a = np.random.rand(3, 4, 5)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.unsqueeze(a_tensor, (0, 2))

    expected_tangent = np.expand_dims(np.ones((3, 4, 5)), axis=(0, 2))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_unsqueeze_tangent_multi_v1():
    a = np.random.rand(2, 3)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.unsqueeze(a_tensor, (1, 3, 4))

    expected_tangent = np.expand_dims(np.ones((2, 3)), axis=(1, 3, 4))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_unsqueeze_tangent_multi_v2():
    a = np.random.rand(5, 6, 7, 8)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((5, 6, 7, 8)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.unsqueeze(a_tensor, (0, 2, 5))

    expected_tangent = np.expand_dims(np.ones((5, 6, 7, 8)), axis=(0, 2, 5))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_unsqueeze_tangent_multi_v3():
    a = np.random.rand(4, 3)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.unsqueeze(a_tensor, (1,))

    expected_tangent = np.expand_dims(np.ones((4, 3)), axis=1)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_unsqueeze_tangent_multi_v4():
    a = np.random.rand(2, 5, 3)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 5, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.unsqueeze(a_tensor, (0, 3))

    expected_tangent = np.expand_dims(np.ones((2, 5, 3)), axis=(0, 3))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_transpose_tangent_multi_v0():
    a = np.random.rand(3, 4, 5)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.transpose(a_tensor, 1, 2)

    expected_tangent = np.swapaxes(np.ones((3, 4, 5)), 1, 2)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_transpose_tangent_multi_v1():
    a = np.random.rand(2, 3)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.transpose(a_tensor, 0, 1)

    expected_tangent = np.swapaxes(np.ones((2, 3)), 0, 1)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_transpose_tangent_multi_v2():
    a = np.random.rand(5, 6, 7, 8)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((5, 6, 7, 8)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.transpose(a_tensor, 2, 3)

    expected_tangent = np.swapaxes(np.ones((5, 6, 7, 8)), 2, 3)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_transpose_tangent_multi_v3():
    a = np.random.rand(4, 3)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.transpose(a_tensor, -1, -2)

    expected_tangent = np.swapaxes(np.ones((4, 3)), -1, -2)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_transpose_tangent_multi_v4():
    a = np.random.rand(2, 5, 3)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 5, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.transpose(a_tensor, 0, 2)

    expected_tangent = np.swapaxes(np.ones((2, 5, 3)), 0, 2)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_permute_tangent_rank2_v0():
    a = np.random.rand(8, 15)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((8, 15)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.permute(a_tensor, (1, 0))

    expected_tangent = np.transpose(np.ones((8, 15)))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_permute_tangent_rank3_v0():
    a = np.random.rand(4, 5, 6)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 5, 6)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.permute(a_tensor, (1, 0, 2))

    expected_tangent = np.transpose(np.ones((4, 5, 6)), (1, 0, 2))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_permute_tangent_rank3_v1():
    a = np.random.rand(70, 15, 512)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((70, 15, 512)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.permute(a_tensor, (2, 1, 0))

    expected_tangent = np.transpose(np.ones((70, 15, 512)), (2, 1, 0))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_permute_tangent_rank4_v0():
    a = np.random.rand(3, 4, 5, 6)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4, 5, 6)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.permute(a_tensor, (3, 2, 1, 0))

    expected_tangent = np.transpose(np.ones((3, 4, 5, 6)), (3, 2, 1, 0))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_permute_tangent_rank4_v1():
    a = np.random.rand(6, 7, 8, 9)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((6, 7, 8, 9)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.permute(a_tensor, (0, 3, 2, 1))

    expected_tangent = np.transpose(np.ones((6, 7, 8, 9)), (0, 3, 2, 1))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_view_tangent_rank1_to_rank2():
    a = np.random.rand(10)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(10)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.view(a_tensor, (2, 5))

    expected_tangent = np.ones(10).reshape(2, 5)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_view_tangent_rank2_to_rank3():
    a = np.random.rand(6, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((6, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.view(a_tensor, (2, 3, 4))

    expected_tangent = np.ones((6, 4)).reshape(2, 3, 4)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_view_tangent_rank3_to_rank4():
    a = np.random.rand(2, 3, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.view(a_tensor, (2, 2, 3, 2))

    expected_tangent = np.ones((2, 3, 4)).reshape(2, 2, 3, 2)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_view_tangent_rank2_to_rank1():
    a = np.random.rand(4, 3)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.view(a_tensor, (12,))

    expected_tangent = np.ones((4, 3)).reshape(12)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_view_tangent_rank3_to_rank2():
    a = np.random.rand(2, 3, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.view(a_tensor, (6, 4))

    expected_tangent = np.ones((2, 3, 4)).reshape(6, 4)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_view_tangent_rank4_to_rank2():
    a = np.random.rand(2, 2, 3, 2)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 2, 3, 2)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.view(a_tensor, (4, 6))

    expected_tangent = np.ones((2, 2, 3, 2)).reshape(4, 6)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_view_tangent_with_negative_dim():
    a = np.random.rand(3, 4, 5)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 4, 5)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.view(a_tensor, (-1, 5))

    expected_tangent = np.ones((3, 4, 5)).reshape(-1, 5)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_reshape_tangent_rank1_to_rank2():
    a = np.random.rand(12)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(12)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.reshape(a_tensor, (3, 4))

    expected_tangent = np.ones(12).reshape(3, 4)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_reshape_tangent_rank2_to_rank3():
    a = np.random.rand(8, 6)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((8, 6)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.reshape(a_tensor, (2, 4, 6))

    expected_tangent = np.ones((8, 6)).reshape(2, 4, 6)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_reshape_tangent_rank3_to_rank4():
    a = np.random.rand(3, 2, 6)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((3, 2, 6)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.reshape(a_tensor, (1, 3, 4, 3))

    expected_tangent = np.ones((3, 2, 6)).reshape(1, 3, 4, 3)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_reshape_tangent_rank2_to_rank1():
    a = np.random.rand(5, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((5, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.reshape(a_tensor, (20,))

    expected_tangent = np.ones((5, 4)).reshape(20)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_reshape_tangent_rank3_to_rank2():
    a = np.random.rand(2, 6, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 6, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.reshape(a_tensor, (12, 4))

    expected_tangent = np.ones((2, 6, 4)).reshape(12, 4)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_reshape_tangent_rank4_to_rank2():
    a = np.random.rand(4, 3, 2, 2)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 3, 2, 2)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.reshape(a_tensor, (6, 8))

    expected_tangent = np.ones((4, 3, 2, 2)).reshape(6, 8)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_reshape_tangent_with_negative_dim():
    a = np.random.rand(4, 5, 3)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 5, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.reshape(a_tensor, (-1, 15))

    expected_tangent = np.ones((4, 5, 3)).reshape(-1, 15)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_clone_tangent_scalar():
    a = np.random.rand()
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(1.0))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.clone(a_tensor)

    expected_tangent = np.ones(1)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_clone_tangent_vector():
    a = np.random.rand(5)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(5)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.clone(a_tensor)

    expected_tangent = np.ones(5)
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_clone_tangent_matrix():
    a = np.random.rand(4, 3)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((4, 3)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.clone(a_tensor)

    expected_tangent = np.ones((4, 3))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_clone_tangent_higher_rank_tensor():
    a = np.random.rand(2, 3, 4)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(
        grad=neuro.tensor(np.ones((2, 3, 4)))
    )
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = neuro.clone(a_tensor)

    expected_tangent = np.ones((2, 3, 4))
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_slice_tangent_single_index():
    a = np.random.rand(10)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(10)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = a_tensor[3]

    expected_tangent = np.ones(10)[3]
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_slice_tangent_range():
    a = np.random.rand(10)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(10)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = a_tensor[2:7]

    expected_tangent = np.ones(10)[2:7]
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_slice_tangent_step():
    a = np.random.rand(10)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(10)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = a_tensor[1:8:2]

    expected_tangent = np.ones(10)[1:8:2]
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_slice_tangent_negative_indices():
    a = np.random.rand(10)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(10)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = a_tensor[-5:-2]

    expected_tangent = np.ones(10)[-5:-2]
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def test_slice_tangent_mixed_indices():
    a = np.random.rand(10)
    a_tensor = neuro.tensor(a, usegrad=True).mutated(grad=neuro.tensor(np.ones(10)))
    with neuro.autograd(enabled=True, reverse=False, forward=True):
        result_tensor = a_tensor[1:-2]

    expected_tangent = np.ones(10)[1:-2]
    np.testing.assert_allclose(result_tensor.grad.data, expected_tangent)


def main():

    # Add Tangent Tests

    test_add_tangent_scalar()
    test_add_tangent_vector()
    test_add_tangent_matrix()

    # Sub Tangent Tests

    test_sub_tangent_scalar()
    test_sub_tangent_vector()
    test_sub_tangent_matrix()

    # Mul Tangent Tests

    test_mul_tangent_scalar()
    test_mul_tangent_vector()
    test_mul_tangent_matrix()

    # Div Tangent Tests

    test_div_tangent_scalar()
    test_div_tangent_vector()
    test_div_tangent_matrix()

    # Dot Tangent Tests

    test_dot_tangent_vector_vector()
    test_dot_tangent_vector_matrix()
    test_dot_tangent_matrix_vector()
    test_dot_tangent_matrix_matrix()

    # Matmul Tangent Tests

    test_matmul_tangent_square_matrices()
    test_matmul_tangent_different_shapes()
    test_matmul_tangent_higher_rank_same_shape()
    test_matmul_tangent_higher_rank_different_shape()
    test_matmul_tangent_different_shapes()

    # Pow Tangent Tests

    test_pow_tangent_scalar()
    test_pow_tangent_vector()
    test_pow_tangent_matrix()

    test_pow_tangent_vector_exp()
    test_pow_tangent_matrix_exp()

    # Exp Tangent Tests

    test_exp_tangent_scalar()
    test_exp_tangent_vector()
    test_exp_tangent_matrix()

    # Log Tangent Tests

    test_log_tangent_scalar()
    test_log_tangent_vector()
    test_log_tangent_matrix()

    # Sin Tangent Tests

    test_sin_tangent_scalar()
    test_sin_tangent_vector()
    test_sin_tangent_matrix()

    # Cos Tangent Tests

    test_cos_tangent_scalar()
    test_cos_tangent_vector()
    test_cos_tangent_matrix()

    # Sum Tangent Tests

    test_sum_tangent_single_dim()
    test_sum_tangent_multiple_dim()
    test_sum_tangent_keepdims()
    test_sum_tangent_single_element_rank1()
    test_sum_tangent_higher_rank_tensor()

    # Max Tangent Tests

    test_max_tangent_single_dim()
    test_max_tangent_multiple_dim()
    test_max_tangent_keepdims()
    test_max_tangent_single_element_rank1()
    test_max_tangent_higher_rank_tensor()

    # Min Tangent Tests

    test_min_tangent_single_dim()
    test_min_tangent_multiple_dim()
    test_min_tangent_keepdims()
    test_min_tangent_single_element_rank1()
    test_min_tangent_higher_rank_tensor()

    # Squeeze Tangent Tests

    test_squeeze_tangent_rank1_v0()
    test_squeeze_tangent_rank1_v1()
    test_squeeze_tangent_rank2_v0()
    test_squeeze_tangent_rank2_v1()

    test_squeeze_tangent_multi_v0()
    test_squeeze_tangent_multi_v1()
    test_squeeze_tangent_multi_v2()

    # Unsqueeze Tangent Tests

    test_unsqueeze_tangent_multi_v0()
    test_unsqueeze_tangent_multi_v1()
    test_unsqueeze_tangent_multi_v2()
    test_unsqueeze_tangent_multi_v3()
    test_unsqueeze_tangent_multi_v4()

    # Transpose Tangent Tests

    test_transpose_tangent_multi_v0()
    test_transpose_tangent_multi_v1()
    test_transpose_tangent_multi_v2()
    test_transpose_tangent_multi_v3()
    test_transpose_tangent_multi_v4()

    # Permute Tangent Tests

    test_permute_tangent_rank2_v0()
    test_permute_tangent_rank3_v0()
    test_permute_tangent_rank3_v1()
    test_permute_tangent_rank4_v0()
    test_permute_tangent_rank4_v1()

    # View Tangent Tests

    test_view_tangent_rank1_to_rank2()
    test_view_tangent_rank2_to_rank3()
    test_view_tangent_rank3_to_rank4()

    test_view_tangent_rank2_to_rank1()
    test_view_tangent_rank3_to_rank2()
    test_view_tangent_rank4_to_rank2()
    test_view_tangent_with_negative_dim()

    # Reshape Tangent Tests

    test_reshape_tangent_rank1_to_rank2()
    test_reshape_tangent_rank2_to_rank3()
    test_reshape_tangent_rank3_to_rank4()

    test_reshape_tangent_rank2_to_rank1()
    test_reshape_tangent_rank3_to_rank2()
    test_reshape_tangent_rank4_to_rank2()
    test_reshape_tangent_with_negative_dim()

    # Clone Tangent Tests

    test_clone_tangent_scalar()
    test_clone_tangent_vector()
    test_clone_tangent_matrix()
    test_clone_tangent_higher_rank_tensor()

    # Slice Tangent Tests

    test_clone_tangent_scalar()
    test_clone_tangent_vector()
    test_clone_tangent_matrix()
    test_clone_tangent_higher_rank_tensor()

    print("All tests passed")


if __name__ == "__main__":
    main()
