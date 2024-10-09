#!/usr/bin/env python

import faulthandler
import warnings
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from geometric_object_classification.tasks import Classification
from geometric_object_classification.task_forecasting import Forecast_Class
from geometric_object_classification.config import instantiate_datamodule, instantiate_model, instantiate_task
from geometric_object_classification.plots import Plots
from geometric_object_classification.utils import (
    WandbModelCheckpoint,
    WandbSummaries,
    get_logger,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)

# Enable faulthandler for debugging segmentation faults
faulthandler.enable(all_threads=False)

# Ignore specific warnings from PyTorch Lightning and WandB
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)

log = get_logger()


def get_callbacks(config):
    monitor = {"monitor": "val/top1", "mode": "max"}
    callbacks = [
        WandbSummaries(**monitor),
        WandbModelCheckpoint(save_last=True, save_top_k=1, every_n_epochs=1, filename="best", **monitor),
        TQDMProgressBar(refresh_rate=1),
        Plots(),
    ]
    if config.early_stopping:
        callbacks.append(EarlyStopping(
            patience=15,
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        ))

    return callbacks


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    """
    Main function for training the model.

    Args:
        config (DictConfig): The train.yaml configuration dictionary.
    """
    set_seed(config)
    OmegaConf.resolve(config)
    print_config(config)

    wandb.init(**config.wandb, resume=(config.wandb.mode == "online") and "allow", notes=config.note)

    log.info("Loading data")
    datamodule = instantiate_datamodule(config.data)
    datamodule.prepare_data()
    datamodule.setup("train")

    log.info("Instantiating model")
    model = instantiate_model(config.model)
    task = instantiate_task(config.task, model, datamodule)

    logger = WandbLogger()
    log_hyperparameters(logger, config, model)

    log.info("Instantiating trainer")
    trainer = instantiate(config.trainer, callbacks=get_callbacks(config), logger=logger)

    log.info("Starting training!")
    trainer.fit(task, datamodule=datamodule)

    if config.save_model:
        checkpoint_path = f"saved_models/model_{config.wandb.name}_db={config.data.db}.ckpt"
        trainer.save_checkpoint(checkpoint_path)

        if config.log_model_to_wandb:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    wandb.finish()
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    return float(trainer.checkpoint_callback.best_model_score or 0)


@hydra.main(config_path="config", config_name="train_continue", version_base=None)
@print_exceptions
def continue_training(config: DictConfig):
    """
    Function for continuing the training of a pre-trained model.

    Args:
        config (DictConfig): The train.yaml configuration dictionary.
    """
    set_seed(config)

    wandb.init(project="Informer", entity="space_debri", id="yehwhckr", resume="allow")

    log.info(f"\n\n\n{config.task.name}\n\n\n")

    model = Forecast_Class.load_from_checkpoint(
        config.pretrained_model_path,
        model=instantiate_model(config.model),
        n_classes=4
    )

    log.info("Loading data")
    datamodule = instantiate_datamodule(config.data)
    datamodule.prepare_data()
    datamodule.setup("train")

    logger = WandbLogger()
    log_hyperparameters(logger, config, model)

    trainer = instantiate(config.trainer, callbacks=get_callbacks(config), logger=logger)

    log.info("Continuing training!")
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.pretrained_model_path)

    if config.save_model:
        checkpoint_path = f"saved_models/continued_model_{config.wandb.name}_db={config.data.db}.ckpt"
        trainer.save_checkpoint(checkpoint_path)

        if config.log_model_to_wandb:
            artifact = wandb.Artifact('continued_model', type='model')
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    if config.eval_testset:
        log.info("Starting testing on continued model!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    wandb.finish()
    log.info(f"Best checkpoint path of continued training:\n{trainer.checkpoint_callback.best_model_path}")
    return float(trainer.checkpoint_callback.best_model_score or 0)


if __name__ == "__main__":
    main()
