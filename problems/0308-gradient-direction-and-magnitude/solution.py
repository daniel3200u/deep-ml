import torch

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient (float)
		- direction: Unit vector (torch.Tensor) in direction of steepest ascent
		- descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
	"""
	# Your code here
	result={}
	result['magnitude']=torch.sqrt(torch.sum(torch.pow(torch.tensor(gradient),2)))
	if result['magnitude'] == 0:
        result['direction'] = torch.zeros(len(gradient))
    else:
        result['direction'] = torch.tensor(gradient) / result['magnitude']
	result['descent_direction']=result['direction']*-1
	return result
	pass