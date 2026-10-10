import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    assert matrix.shape==(2,2),"error"
    eigenvalues = torch.linalg.eigvals(matrix)
    sorted_eigenvalues = torch.sort(eigenvalues.real,descending=True).values
    return sorted_eigenvalues
    pass
