import torch
import numpy as np

def apply_subsample_signal(data, section_length_percentage):
    '''
    Zeroes out a random contiguous section of the signal based on a percentage of the total length.
    This operation is applied across all examples in the batch consistently.

    Parameters:
    data (torch.Tensor): The input data tensor with shape [batch_size, num_time_steps, ...].
    section_length_percentage (float): The percentage of the total length to zero out.

    Returns:
    torch.Tensor: The modified data tensor with a section zeroed out.
    '''
    if section_length_percentage < 0 or section_length_percentage > 1:
        raise ValueError("Section length percentage must be between 0 and 1.")

    num_time_steps = data.shape[1]
    section_length = int(num_time_steps * section_length_percentage)
    
    if section_length >= num_time_steps:
        raise ValueError("Section length must be smaller than the length of the data.")
    
    if section_length == 0:
        return data
    
    # Choose a random start index for the section to zero out
    start_index = np.random.randint(0, num_time_steps - section_length + 1)

    # Zero out the selected section across all examples in the batch
    data_clone = data.clone()  # Create a copy to avoid modifying the original data
    data_clone[:, start_index:start_index + section_length, ...] = 0
    
    return data_clone
