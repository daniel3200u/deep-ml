import torch

def descriptive_statistics(data) -> dict:
    # PERBAIKAN 1: Paksa jadi float sejak awal agar quantile tidak error
    data_t = torch.as_tensor(data, dtype=torch.float32)
    
    result = {}
    result['mean'] = torch.mean(data_t).item()
    result['median'] = torch.quantile(data_t, 0.5).item()
    result['mode'] = int(torch.mode(data_t).values.item())
    result['variance'] = torch.var(data_t, unbiased=False).item()
    result['standard_deviation'] = torch.std(data_t, unbiased=False).item()
    quantiles = torch.quantile(data_t, torch.tensor([0.25, 0.50, 0.75]))
    result['25th_percentile'] = quantiles[0].item()
    result['50th_percentile'] = quantiles[1].item()
    result['75th_percentile'] = quantiles[2].item()
    result['interquartile_range'] = (quantiles[2] - quantiles[0]).item()
    
    return result