
import numpy as np
import torch

def apply_occlusion_random_dropout(data, intensity):
    '''
    Zeroes out a certain percentage of time steps in the data.
    
    Parameters:
    data (torch.Tensor): The input data tensor.
    intensity (float): The percentage of time steps to zero out.

    Returns:
    torch.Tensor: The modified data tensor.
    '''
    data_clone = data.clone()
    num_time_steps = data_clone.shape[1]
    num_to_zero_out = int(num_time_steps * intensity)
    
    # Randomly selecting time steps to zero out
    zero_indices = np.random.choice(num_time_steps, num_to_zero_out, replace=False)
    data_clone[:, zero_indices] = 0
    
    return data_clone

def apply_patterned_occlusion(data, occlusion_pattern):
    # ... (rest of your function)
    # Example: Sinusoidal Pattern
    time_steps = np.arange(data.shape[1])
    occlusion_pattern = amplitude * np.sin(2 * np.pi * frequency * time_steps + phase)
    data_clone += occlusion_pattern
    # ... (rest of your function)
    
import numpy as np
import torch

def apply_sinusoidal_occlusion(data, amplitude, frequency, phase):
    '''
    Applies sinusoidal occlusion by reducing the signal strength based on a sinusoidal pattern.

    Parameters:
    data (torch.Tensor): The input data tensor.
    amplitude (float): The amplitude of the sinusoidal occlusion, representing the maximum reduction in signal.
    frequency (float): The frequency of the sinusoidal occlusion, representing how often the signal is occluded.
    phase (float): The phase shift of the sinusoidal occlusion, determining the starting point of the occlusion pattern.

    Returns:
    torch.Tensor: The modified data tensor with sinusoidal occlusion applied.
    '''
    data_clone = data.clone()
    num_time_steps = data_clone.shape[1]
    t = np.linspace(0, 1, num_time_steps)  # Normalized time vector
    
    # Create the sinusoidal occlusion pattern
    occlusion_pattern = amplitude * (1 + np.sin(2 * np.pi * frequency * t + phase)) / 2  # Normalize to range [0, amplitude]
    
    # Apply the occlusion pattern to the data
    data_clone -= data_clone * occlusion_pattern
    
    return data_clone
