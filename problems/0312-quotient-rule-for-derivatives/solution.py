import torch
def eval_poly(coeffs, x):
    # coeffs = [1, 0, 1]  ->  1*x^2 + 0*x^1 + 1*x^0
    result = 0
    n = len(coeffs)
    for i, c in enumerate(coeffs):
        power = n - 1 - i     # <-- kunci: pangkat menurun
        result += c * (x ** power)
    return result
def derivative_coeffs(coeffs):
    n = len(coeffs)
    if n <= 1:
        return [0]         # turunan konstanta = 0
    result = []
    for i, c in enumerate(coeffs):
        power = n - 1 - i
        # turunan c*x^power = c*power*x^(power-1)
        result.append(c * power)
    return result
def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> torch.Tensor:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x) as a scalar torch.Tensor
    """
    # Your code here
    x_t = torch.tensor(x, requires_grad=True)
    g = eval_poly(g_coeffs, x_t)
    h = eval_poly(h_coeffs, x_t)

    # Ambil turunan
    g.backward(retain_graph=True)     # <-- retain_graph penting kalau backward 2x
    g_prime = x_t.grad.item()
    x_t.grad = None                   # reset gradien

    h.backward()
    h_prime = x_t.grad.item()

    result = (g_prime * h.item() - g.item() * h_prime) / (h.item() ** 2)
    return torch.tensor(result)
    pass