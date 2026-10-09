import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here
    if norm_type=='l1':
        abs_mat=torch.abs(arr)
        total=torch.sum(abs_mat)
    elif norm_type=='l2':
        abs_mat=torch.sum(torch.pow(torch.abs(arr),2))
        total=torch.sqrt(abs_mat)
    elif norm_type=='linf':
        abs_mat=torch.max(torch.abs(arr))
        total=abs_mat
    elif norm_type=='frobenius' and arr.dim()==2:
        total = torch.sqrt(torch.sum(arr ** 2))
    else:
        raise ValueError('ValueError')
    return float(total)
    pass
