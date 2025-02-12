import pytest
import numpy as np
    
from src.base.dl_math import deriv 


def test_deriv():
    # Define a simple function: f(x) = x^2, derivative should be 2x
    func = lambda x: x**2
    input_ = np.array([1.0, 2.0, 3.0])
    expected_derivative = 2 * input_  # Analytical derivative: 2x
    
    computed_derivative = deriv(func, input_)
    
    # Check if computed derivative is close to the expected value
    np.testing.assert_allclose(computed_derivative, expected_derivative, rtol=1e-3)

if __name__ == "__main__":
    pytest.main()
