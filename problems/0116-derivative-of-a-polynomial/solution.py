import torch

def poly_term_derivative(c: float, x: float, n: float) -> torch.Tensor:
    """
    Compute the derivative of a polynomial term c * x^n at point x.
    
    Args:
        c: coefficient of the term
        x: point at which to evaluate the derivative
        n: exponent of the term
    
    Returns:
        The value of the derivative at point x as a tensor
    """
    c = torch.tensor(c)
    x = torch.tensor(x)
    n = torch.tensor(n)
    return c * n * (x ** (n - 1))
    pass