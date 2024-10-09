from pathlib import Path
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pytorch_lightning as pl
import os
import json
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import wandb

class BadRadarDataset(Dataset):
    def __init__(self, data_dir="config/data/stretch_test_set_el_0/noise", transform=None, db=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        self.db=db
        
        # Initialize counts
        class_samples_count = {'cylinders': 0, 'cones': 0}
        zero_value_samples = {'cylinders': 0, 'cones': 0}
        
        # class_id_mapping = {'cylinders': -1, 'cones': -2}
        class_id_mapping = {'cones': -2}
        
        for class_name, class_id in class_id_mapping.items():
            class_path = self.data_dir / class_name
            for i in range(1, 2):
                file_name = f'object_h1_r005_noise_{i}.mat'
                file_path = class_path / file_name
                
                if file_path.exists():
                    data = scipy.io.loadmat(file_path)['objects_echo_noise']
                    for sample in data:
                        
                        tensor_sample = torch.tensor(sample, dtype=torch.float32)
                        if not (torch.isnan(tensor_sample).any() or torch.isinf(tensor_sample).any()):
                            if torch.all(tensor_sample == 0):
                                zero_value_samples[class_name] += 1
                            else:       
                                class_samples_count[class_name] += 1
                                self.samples.append(tensor_sample)
                                self.labels.append(class_id)
                        else:
                            print(f"NaN or Inf found in sample from file: {file_path}")
                else:
                    print(f"File not found: {file_path}")

        # Printing information
        print(f"Total samples loaded: {len(self.samples)}")
        for shape, count in class_samples_count.items():
            print(f"Total samples for {shape}: {count}")
        for shape, count in zero_value_samples.items():
            print(f"Samples with only 0.0 values for {shape}: {count}")

        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        
        if self.db: 
            # Convert to squared values
            sample_squared = torch.pow(sample, 2)

            # Convert to decibels
            sample_db = 10 * torch.log10(sample_squared + 1e-10)  # Adding a small value to avoid log(0)
            
            if self.transform:
                return self.transform(sample_db).unsqueeze(0), label
            
            return sample_db.unsqueeze(0), label
        else:
            if self.transform:
                return self.transform(sample).unsqueeze(0), label
            
            return sample.unsqueeze(0), label
    
    def get_transformed_sample(self, idx):
        """
        Get the original, squared, dB converted, and normalized samples.
        """
        original_sample = self.samples[idx]
        squared_sample = torch.pow(original_sample, 2)
        sample_db = 10 * torch.log10(squared_sample + 1e-10)  # dB conversion

        if self.transform:
            normalized_sample = self.transform(sample_db)
        else:
            normalized_sample = original_sample  # If no transform, return original

        return {
            "original": original_sample,
            "squared": squared_sample,
            "db": sample_db,
            "normalized": normalized_sample
        }
        