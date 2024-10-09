#!/usr/bin/env python

import faulthandler
import warnings
import hydra
import wandb
import pickle
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from geometric_object_classification.config import instantiate_datamodule, instantiate_model
from geometric_object_classification.tasks import Classification
from geometric_object_classification.robustness_data_modifications.artificial_cluster_noise import apply_clutter_noise
from geometric_object_classification.robustness_data_modifications.artificial_occlusion import apply_occlusion_random_dropout
from geometric_object_classification.robustness_data_modifications.sensor_saturation import apply_percentile_saturation
from geometric_object_classification.robustness_data_modifications.subsample_signal import apply_subsample_signal
from geometric_object_classification.data import NormalizeTransform, calculate_dataset_stats
from geometric_object_classification.utils import (
    get_logger,
    print_config,
    print_exceptions,
)

faulthandler.enable(all_threads=False)
warnings.filterwarnings("ignore", "There is a wandb run already in progress", module="pytorch_lightning.loggers.wandb")

log = get_logger()

def apply_modifications(config, intensity, modify, after_transform=False):
    """
    Applies data modification with optional transformations.
    """
    modified_test_set = []
    datamodule = instantiate_datamodule(config.data)
    datamodule.db = not after_transform
    datamodule.normalize = not after_transform
    test_set = prepare_test_dataset(datamodule)

    if after_transform:
        for data, label in test_set:
            modified_sample = modify_and_validate(data, intensity)
            modified_test_set.append((normalize_and_convert_to_db(modified_sample), label))

        mean, std = calculate_dataset_stats(modified_test_set)
        normalize_transform = NormalizeTransform(mean, std)
        modified_test_set = [(normalize_transform(sample), label) for sample, label in modified_test_set]
    else:
        for data, label in test_set:
            modified_test_set.append((modify(data, intensity), label))

    return modified_test_set

def modify_and_validate(data, intensity):
    """
    Modifies the data and checks for NaN or inf values.
    """
    modified_sample = apply_modifications(data, intensity)
    if torch.isnan(modified_sample).any() or torch.isinf(modified_sample).any():
        raise RuntimeError("NaN or inf detected during modification!")
    return modified_sample

def normalize_and_convert_to_db(sample):
    """
    Converts data to dB scale and normalizes.
    """
    sample_squared = torch.pow(sample, 2)
    sample_db = 10 * torch.log10(sample_squared + 1e-10)
    if torch.isnan(sample_db).any() or torch.isinf(sample_db).any():
        raise RuntimeError("NaN or inf detected during dB conversion!")
    return sample_db

def prepare_test_dataset(datamodule):
    """
    Prepares the test dataset.
    """
    datamodule.prepare_data()
    datamodule.setup(stage='test')
    return datamodule.test_data

def save_test_dataset_to_pickle(test_dataset):
    """
    Saves the test dataset to a pickle file.
    """
    with open('test_dataset.pkl', 'wb') as file:
        pickle.dump(test_dataset, file)

def test_modifications(config, model, modify_func, intensities, description):
    """
    Tests model performance on modified datasets.
    """
    log.info(f"Deploy {description}")
    table_acc = wandb.Table(columns=["intensity", "accuracy"])

    for intensity in intensities:
        trainer = Trainer()
        modified_dataset = apply_modifications(config, intensity, modify_func)
        test_data_loader = DataLoader(modified_dataset, batch_size=128)
        result = trainer.test(model, dataloaders=test_data_loader)
        table_acc.add_data(intensity, result[0]['test/accuracy'])

    wandb.log({f"{description} effect on accuracy": table_acc})

def test_occlusion_random_dropouts(config, model):
    test_modifications(config, model, apply_occlusion_random_dropout, np.linspace(0.0, 0.8, 9), "random dropouts")

def test_subsampling(config, model):
    test_modifications(config, model, apply_subsample_signal, np.linspace(0.0, 0.8, 9), "subsampling")

def test_saturation(config, model):
    log.info("Deploy test_saturation")
    max_percentiles = np.linspace(1.0, 0.1, 10)
    table_acc = wandb.Table(columns=["saturation_cut_off_percentile", "accuracy"])

    for max_percentile in max_percentiles:
        trainer = Trainer()
        modified_dataset = apply_modifications(config, max_percentile, apply_percentile_saturation, after_transform=True)
        test_data_loader = DataLoader(modified_dataset, batch_size=128)
        result = trainer.test(model, dataloaders=test_data_loader)
        table_acc.add_data((1.0 - max_percentile), result[0]['test/accuracy'])

    wandb.log({"percentile saturation effect on accuracy": table_acc})

def test_cluster_sinusoid(config, model):
    log.info("Deploy test_cluster_sinusoid")
    radar_freq = 94e9
    frequencies = [
        radar_freq * 1.05, radar_freq - 1e9, radar_freq / 2, 1e9, radar_freq * 2
    ]
    SNR_dB = np.array([0.0, 2.5, 5.0, 10.0, 15.0, 20.0])
    table_acc = wandb.Table(columns=["intensity", "accuracy", "frequency"])

    for freq in frequencies:
        for snr in SNR_dB:
            trainer = Trainer()
            modify_func = lambda data, _: apply_clutter_noise(data, None, desired_SNR_dB=snr, clutter_type="sinusoidal", frequency=freq)
            modified_dataset = apply_modifications(config, None, modify_func, after_transform=True)
            test_data_loader = DataLoader(modified_dataset, batch_size=128)
            result = trainer.test(model, dataloaders=test_data_loader)
            table_acc.add_data(snr, result[0]['test/accuracy'], freq)

    wandb.log({"cluster sinusoid effect on accuracy": table_acc})

def test_cluster_random_peaks(config, model):
    test_modifications(config, model, lambda data, intensity: apply_clutter_noise(data, intensity, clutter_type="high_peaks"), 
                       np.array([0.0, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2]), "high peaks")

@hydra.main(config_path="config", config_name="test_robustness", version_base=None)
@print_exceptions
def main(config: DictConfig):
    OmegaConf.resolve(config)
    print_config(config)

    model_path = get_model_path(config)
    wandb.init(**config.wandb, resume=(config.wandb.mode == "online") and "allow", notes=config.note)

    model_architecture = instantiate_model(config.model)
    model = Classification.load_from_checkpoint(model_path, model=model_architecture, n_classes=4)

    test_method_map = {
        'random_dropouts': test_occlusion_random_dropouts,
        'subsample': test_subsampling,
        'cluster_sinusoid': test_cluster_sinusoid,
        'percentile_saturation': test_saturation,
        'cluster_random_peaks': test_cluster_random_peaks,
    }

    test_method = config.get('test_method', 'normal')
    log.info(f"Running test method: {test_method}")

    if test_method in test_method_map:
        test_method_map[test_method](config, model)

    wandb.finish()

def get_model_path(config):
    """
    Gets the model path based on the configuration.
    """
    model_type = config.model.name
    if model_type == "lstm":
        config.wandb.name = f"lstm_{'bidirectional' if config.model.bidirectional else 'vanilla'}_run_{config.iteration}"
        config.wandb.group = f"{'bidirectional' if config.model.bidirectional else 'vanilla'}_lstm"
        return config.model_paths.lstm_bidirectional if config.model.bidirectional else config.model_paths.lstm_vanilla
    return config['model_paths'][model_type]

if __name__ == '__main__':
    main()
