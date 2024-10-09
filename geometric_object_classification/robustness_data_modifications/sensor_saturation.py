import torch

def apply_percentile_saturation(data, max_percentile, min_percentile=None):
    '''
    Simulates sensor saturation by capping the data at specified saturation percentiles.

    Parameters:
    data (torch.Tensor): The input data tensor.
    saturation_percentiles (dict): A dictionary containing 'max' and 'min' percentile values for saturation.

    Returns:
    torch.Tensor: The modified data tensor with sensor saturation applied.
    '''
    
    data_clone = data.clone()
    
    # Compute the saturation levels based on percentiles
    if max_percentile is not None:
        if max_percentile < 0.0 or max_percentile > 1.0:
            raise ValueError("Low percentile must be between 0 and 100.")
        
        if max_percentile == 1.0:
            return data
        
        max_saturation = torch.quantile(data, max_percentile)
        data_clone[data_clone > max_saturation] = max_saturation
    
    if min_percentile is not None:
        if (min_percentile < 0 or min_percentile > 1) and min_percentile < max_percentile:
            raise ValueError("Low percentile must be between 0 and 100.")
        
        min_saturation = torch.quantile(data, min_percentile)
        data_clone[data_clone < min_saturation] = min_saturation

    return data_clone

# Example usage of the function
# saturation_percentiles = {'max': 95, 'min': 5}  # Define saturation levels in percentiles
# saturated_data = apply_percentile_saturation(data, saturation_percentiles)

def apply_absolute_threshold_saturation(data, max_saturation, min_saturation=None):
    '''
    Simulates sensor saturation by capping the data at specified saturation levels.

    Parameters:
    data (torch.Tensor): The input data tensor.
    saturation_level (dict): A dictionary containing 'max' and 'min' values for saturation levels.

    Returns:
    torch.Tensor: The modified data tensor with sensor saturation applied.
    '''
    data_clone = data.clone()
    
    if max_saturation is not None:
        
        if max_saturation == 1.0:
            return data_clone
        
        if max_saturation < 0 or max_saturation > 1:
            raise ValueError("Low percentile must be between 0 and 100.")
        
        data_clone[data_clone > max_saturation] = max_saturation
    
    if min_saturation is not None:
        if (min_saturation < 0 or min_saturation > 1) and min_saturation < min_saturation:
            raise ValueError("Low percentile must be between 0 and 100.")
        
        data_clone[data_clone < min_saturation] = min_saturation

    return data_clone

## Example usage of the function
# saturation_level = {'max': 100, 'min': -100}  # Define saturation levels
# saturated_data = apply_sensor_saturation(data, saturation_level)
