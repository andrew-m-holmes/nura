import numpy as np
import deepnet
import deepnet.functional as f

def test_add_jvp_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    result_tensor = f.add(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h + b - (a - h + b)) / (2 * h)
    expected_b = (a + b + h - (a + b - h)) / (2 * h)
    expected = expected_a + expected_b
    if result_tensor.tangent is None:
        print(a_tensor)
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_add_jvp_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones(5)))
    result_tensor = f.add(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h + b - (a - h + b)) / (2 * h)
    expected_b = (a + b + h - (a + b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_add_jvp_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 3))))
    result_tensor = f.add(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h + b - (a - h + b)) / (2 * h)
    expected_b = (a + b + h - (a + b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_sub_jvp_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    result_tensor = f.sub(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h - b - (a - h - b)) / (2 * h)
    expected_b = (a - (b + h) - (a - (b - h))) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_sub_jvp_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones(5)))
    result_tensor = f.sub(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h - b - (a - h - b)) / (2 * h)
    expected_b = (a - (b + h) - (a - (b - h))) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_sub_jvp_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 3))))
    result_tensor = f.sub(a_tensor, b_tensor)

    h = 1e-8
    expected_a = (a + h - b - (a - h - b)) / (2 * h)
    expected_b = (a - (b + h) - (a - (b - h))) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_mul_jvp_scalar():
    a = np.random.rand()
    b = np.random.rand()

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    result_tensor = f.mul(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) * b - (a - h) * b) / (2 * h)
    expected_b = (a * (b + h) - a * (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_mul_jvp_vector():
    a = np.random.rand(5)
    b = np.random.rand(5)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones(5)))
    result_tensor = f.mul(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) * b - (a - h) * b) / (2 * h)
    expected_b = (a * (b + h) - a * (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_mul_jvp_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 3))))
    result_tensor = f.mul(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) * b - (a - h) * b) / (2 * h)
    expected_b = (a * (b + h) - a * (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_div_jvp_scalar():
    a = np.random.rand()   
    b = np.random.rand() 

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(1.))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(1.))
    result_tensor = f.div(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_b = (a / (b + h) - a / (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_div_jvp_vector():
    a = np.random.rand(5) 
    b = np.random.rand(5) 

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones(5)))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones(5)))
    result_tensor = f.div(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_b = (a / (b + h) - a / (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)

def test_div_jvp_matrix():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    a_tensor = deepnet.tensor(a).dual(deepnet.tensor(np.ones((3, 3))))
    b_tensor = deepnet.tensor(b).dual(deepnet.tensor(np.ones((3, 3))))
    result_tensor = f.div(a_tensor, b_tensor)

    h = 1e-8
    expected_a = ((a + h) / b - (a - h) / b) / (2 * h)
    expected_b = (a / (b + h) - a / (b - h)) / (2 * h)
    expected = expected_a + expected_b
    np.testing.assert_allclose(
        result_tensor.tangent.data, expected, rtol=1e-5, atol=1e-5)


def main():


    with deepnet.forward_ad():

        # Add JVP Tests

        test_add_jvp_scalar()
        test_add_jvp_vector()
        test_add_jvp_matrix()

        # Sub JVP Tests

        test_sub_jvp_scalar()
        test_sub_jvp_vector()
        test_sub_jvp_matrix()

        # Mul JVP Tests

        test_mul_jvp_scalar()
        test_mul_jvp_vector()
        test_mul_jvp_matrix()

        # Div JVP Tests

        test_div_jvp_scalar()
        test_div_jvp_vector()
        test_div_jvp_matrix()


        print("All tests passed")

if __name__ == "__main__":
    main()