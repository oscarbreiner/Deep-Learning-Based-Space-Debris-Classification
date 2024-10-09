import numpy as np
import torch

def apply_clutter_noise(data, clutter_intensity=0, clutter_type='gaussian', **kwargs):
    '''
    Adds clutter noise to the dataset to simulate a cluttered environment.

    Parameters:
    data (torch.Tensor): The input data tensor.
    clutter_intensity (float): The intensity or magnitude of the clutter noise.
    clutter_type (str): The type of clutter noise - 'gaussian', 'uniform', 'sinusoidal', etc.
    kwargs: Additional arguments specific to the type of clutter noise.

    Returns:
    torch.Tensor: The modified data tensor with added clutter.
    '''
    data_clone = data.clone()
    noise = None

    if clutter_type == 'gaussian':
        mean = kwargs.get('mean', 0)
        std = kwargs.get('std', 1)
        noise = torch.normal(mean, std, size=data.shape) * clutter_intensity
        data_clone += noise
        
    elif clutter_type == 'uniform':
        low = kwargs.get('low', -1)
        high = kwargs.get('high', 1)
        noise = (high - low) * torch.rand(data.shape) + low
        noise *= clutter_intensity
        data_clone += noise
        
    elif clutter_type == 'sinusoidal':
        # Calculate signal power (assuming data is in volts and in 1 x 501 format)
        signal_power = torch.mean(data_clone ** 2)
        
        desired_SNR_dB = kwargs.get('desired_SNR_dB', 30)  # Default value of 30 dB
        frequency = kwargs.get('frequency', 1)
        phase = kwargs.get('phase', 0)
        
        # Convert SNR from dB to linear scale
        desired_SNR_linear = 10 ** (desired_SNR_dB / 10)
        
        # Calculate noise power for the desired SNR
        noise_power = signal_power / desired_SNR_linear
        
        # Calculate amplitude of the sinusoidal noise (assuming noise is zero-mean)
        amplitude = torch.sqrt(2 * noise_power)
        
        # Generate sinusoidal noise
        time_steps = torch.arange(data.shape[1])
        noise = amplitude * torch.sin(2 * np.pi * frequency * time_steps + phase)
        noise = noise.repeat(data.shape[0], 1)  # Repeat the pattern for all data samples
        data_clone += noise

    elif clutter_type == 'high_peaks':
        # Calculate the number of peaks based on the intensity and the length of the time series
        num_peaks = int(clutter_intensity * data.shape[1])
        
        for i in range(data.shape[0]):
            # Randomly choose indices for the peaks
            peak_indices = np.random.choice(data.shape[1], num_peaks, replace=False)
            
            # Generate peak heights as a NumPy array
            peak_heights_np = np.random.uniform(low=2, high=4, size=num_peaks)
            
            # Convert peak heights to a PyTorch tensor and scale by the maximum value of the data sample
            peak_heights_tensor = torch.from_numpy(peak_heights_np).to(data_clone.dtype).to(data_clone.device)
            peak_heights_scaled = peak_heights_tensor * data_clone[i].max()
            
            # Add the scaled peak heights to the selected indices of the data sample
            data_clone[i, peak_indices] += peak_heights_scaled

    else:
        raise ValueError("Unsupported clutter type. Choose 'gaussian', 'uniform', or 'sinusoidal'.")

    return data_clone
