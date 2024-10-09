#!/usr/bin/env python

import os
import json
import pickle
from pathlib import Path
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit
import wandb

class NormalizeTransform:
    """
    Transform to normalize tensor samples using provided mean and standard deviation.
    Raises an error if standard deviation is zero or if NaN/inf values are detected.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        if self.std == 0:
            raise RuntimeError("Division by zero detected: Standard deviation is zero.")
        norm = (sample - self.mean) / self.std
        if torch.isnan(norm).any():
            raise RuntimeError("NaN detected during normalization!")
        elif torch.isinf(norm).any():
            raise RuntimeError("Inf detected during normalization!")
        return norm

class RadarDataset(Dataset):
    """
    Dataset for radar data classification. Loads data from .mat files, applies optional transformation, and supports
    limiting the number of samples per class.
    """
    def __init__(self, data_dir, transform=None, max_samples_per_class=None, db=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        self.db = db
        self.max_samples_per_class = max_samples_per_class

        class_samples_count = {'cylinders': 0, 'spheres': 0, 'circular_plates': 0, 'cones': 0}
        zero_value_samples = {'cylinders': 0, 'spheres': 0, 'circular_plates': 0, 'cones': 0}

        for class_id, class_name in enumerate(['cylinders', 'spheres', 'circular_plates', 'cones']):
            class_path = self.data_dir / class_name
            for i in range(1, 33):
                file_path = class_path / f'object_noise_{i}.mat'
                
                if file_path.exists():
                    data = scipy.io.loadmat(file_path)['objects_echo_noise']
                    for sample in data:
                        if self.max_samples_per_class is not None and class_samples_count[class_name] >= self.max_samples_per_class:
                            break
                        
                        tensor_sample = torch.tensor(sample, dtype=torch.float32)
                        if not (torch.isnan(tensor_sample).any() or torch.isinf(tensor_sample).any()):
                            if torch.all(tensor_sample == 0):
                                zero_value_samples[class_name] += 1
                            else:       
                                class_samples_count[class_name] += 1
                                self.samples.append(tensor_sample)
                                self.labels.append(class_id)
                else:
                    print(f"File not found: {file_path}")

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
            sample_squared = torch.pow(sample, 2)
            sample_db = 10 * torch.log10(sample_squared + 1e-10)  # Avoid log(0)
            sample = self.transform(sample_db).unsqueeze(0) if self.transform else sample_db.unsqueeze(0)
        else:
            sample = self.transform(sample).unsqueeze(0) if self.transform else sample.unsqueeze(0)

        return sample, label
    
    def get_transformed_sample(self, idx):
        """
        Returns original, squared, dB converted, and normalized versions of a sample.
        """
        original_sample = self.samples[idx]
        squared_sample = torch.pow(original_sample, 2)
        sample_db = 10 * torch.log10(squared_sample + 1e-10)

        normalized_sample = self.transform(sample_db) if self.transform else original_sample

        return {
            "original": original_sample,
            "squared": squared_sample,
            "db": sample_db,
            "normalized": normalized_sample
        }

def plot_samples(dataset):
    """
    Plots one sample for each class in different transformations using wandb.
    """
    class_names = ['cylinders', 'spheres', 'circular_plates', 'cones']
    class_found = {class_name: False for class_name in class_names}
    columns = ["Time_step", "Sample", "Class"]
    tables = {key: wandb.Table(columns=columns) for key in ["original", "squared", "db", "normalized"]}

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_name = class_names[label]
        if not class_found[class_name]:
            sample = dataset.dataset.get_transformed_sample(idx)
            for key in tables:
                for i, y in enumerate(sample[key]):
                    tables[key].add_data(i, y, class_name)
            class_found[class_name] = True

            if all(class_found.values()):
                break

    for key, table in tables.items():
        wandb.log({f"Radar Samples - {key.capitalize()}": table})

def calculate_dataset_stats(dataset):
    """
    Calculates mean and standard deviation for the dataset.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    mean, std = 0.0, 0.0
    for data, _ in loader:
        mean += data.mean()
        std += data.std()
    mean /= len(loader)
    std /= len(loader)
    return mean.item(), std.item()

def save_stats_to_file(stats, filename):
    """
    Saves dataset statistics to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(stats, f)

def load_stats_from_file(filename):
    """
    Loads dataset statistics from a JSON file if it exists.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

class RadarDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling radar dataset, including normalization and dataset splits.
    """
    def __init__(self, root, stats_file, split_seed=80672983, batch_size=128, max_samples_per_class=None, db=True, normalize=True):
        super().__init__()
        self.root = Path(root)
        self.stats_file = stats_file
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.db = db
        self.normalize = normalize
        self.max_samples_per_class = max_samples_per_class
        self.n_classes = 4
        self.image_shape = (1, 501)

    def save_split_indices(self, indices, filename):
        with open(filename, 'wb') as f:
            pickle.dump(indices, f)

    def load_split_indices(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return None

    def print_class_distribution(self, dataset_subset, subset_name=""):
        class_counts = {0: 'cylinders', 1: 'spheres', 2: 'circular_plates', 3: 'cones'}
        counts = {class_name: 0 for class_name in class_counts.values()}
        for _, label in dataset_subset:
            counts[class_counts[label]] += 1

        print(f"Class distribution in {subset_name} set:")
        for class_name, count in counts.items():
            print(f"{class_name}: {count}")

    def setup(self, stage=None):
        dataset = RadarDataset(self.root, max_samples_per_class=self.max_samples_per_class, db=self.db)
        labels = np.array(dataset.labels)
        split_indices_file = 'split_indices.pkl'
        indices = self.load_split_indices(split_indices_file)

        if indices is None:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=self.split_seed)
            train_idx, test_val_idx = next(sss.split(np.zeros(len(labels)), labels))

            labels_for_second_split = labels[test_val_idx]
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=self.split_seed)
            test_idx, val_idx = next(sss_val.split(np.zeros(len(labels_for_second_split)), labels_for_second_split))

            test_idx, val_idx = test_val_idx[test_idx], test_val_idx[val_idx]
            indices = {'train': train_idx, 'test': test_idx, 'val': val_idx}
            self.save_split_indices(indices, split_indices_file)
        else:
            train_idx, test_idx, val_idx = indices['train'], indices['test'], indices['val']

        self.train_data = Subset(dataset, train_idx)
        self.test_data = Subset(dataset, test_idx)
        self.val_data = Subset(dataset, val_idx)

        self.print_class_distribution(self.test_data, "test")
        self.print_class_distribution(self.train_data, "train")
        self.print_class_distribution(self.val_data, "val")

        if self.normalize:
            stats = load_stats_from_file(self.stats_file) or calculate_dataset_stats(self.train_data)
            if isinstance(stats, tuple):
                stats = {'mean': stats[0], 'std': stats[1]}
                save_stats_to_file(stats, self.stats_file)

            normalize_transform = NormalizeTransform(stats['mean'], stats['std'])
            self.train_data.dataset.transform = normalize_transform
            self.val_data.dataset.transform = normalize_transform
            self.test_data.dataset.transform = normalize_transform

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

if __name__ == "__main__":
    wandb.init(project="data_plot", group="lines2", name="lines")
    data_module = RadarDataModule(
        root="config/data/noise",
        stats_file="stats.json",
        batch_size=128, max_samples_per_class=None, db=True
    )
    data_module.setup()
    plot_samples(data_module.test_data)
    wandb.finish()
