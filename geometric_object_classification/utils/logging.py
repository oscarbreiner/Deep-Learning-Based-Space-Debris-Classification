import inspect
import logging

import pytorch_lightning as pl
import rich
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
import json


def get_logger():
    caller = inspect.stack()[1]
    module = inspect.getmodule(caller.frame)
    logger_name = None
    if module is not None:
        logger_name = module.__name__.split(".")[-1]
    return logging.getLogger(logger_name)


@rank_zero_only
def print_config(config: DictConfig) -> None:
    content = OmegaConf.to_yaml(config, resolve=True)
    rich.print(Syntax(content, "yaml"))


def count_params(model: nn.Module):
    return {
        "params-total": sum(p.numel() for p in model.parameters()),
        "params-trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "params-not-trainable": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
    }


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@rank_zero_only
def log_hyperparameters(logger: pl.loggers.Logger, config: DictConfig, model: pl.LightningModule):
    # Convert OmegaConf to a regular dictionary and flatten it
    hparams = flatten_dict(OmegaConf.to_container(config, resolve=True))

    # Update model parameters
    hparams.setdefault("model", {}).update(count_params(model))

    # Serialize complex objects (like lists) to string for better logging
    for key, value in hparams.items():
        if isinstance(value, (list, dict)):
            hparams[key] = json.dumps(value)

    # Log the hyperparameters
    logger.log_hyperparams(hparams)

    # Disable further logging of hyperparameters to avoid duplication
    logger.log_hyperparams = lambda *args, **kwargs: None
