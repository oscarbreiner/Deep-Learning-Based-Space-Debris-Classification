#!/usr/bin/env python

import faulthandler
import warnings

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
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

# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

# Stop pytorch-lightning from pestering us about things we already know
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)
# If data loading is really not a bottleneck for you, you uncomment this to silence the
# warning about it
# warnings.filterwarnings(
#     "ignore",
#     "The dataloader, [^,]+, does not have many workers",
#     module="pytorch_lightning",
# )

log = get_logger()


def get_callbacks(config):
    monitor = {"monitor": "val/top1", "mode": "max"}
    # monitor = {"monitor": "val/mse", "mode": "max"}
    callbacks = [
        WandbSummaries(**monitor),
        WandbModelCheckpoint(
            save_last=True, save_top_k=1, every_n_epochs=1, filename="best", **monitor
        ),
        TQDMProgressBar(refresh_rate=1),
        Plots(),
    ]
    if config.early_stopping is not None:
        # config.early_stopping
        stopper = EarlyStopping(
            patience=int(15),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)

    return callbacks


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    rng = set_seed(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    print_config(config)
    wandb.init(**config.wandb, resume=(config.wandb.mode == "online") and "allow", notes=config.note)

    


    log.info("Loading data")
    datamodule = instantiate_datamodule(config.data, rng)
    datamodule.prepare_data()
    datamodule.setup("train")

    log.info("Instantiating model")
    model = instantiate_model(config.model)

    task = instantiate_task(config.task, model, datamodule)

    logger = WandbLogger()
    log_hyperparameters(logger, config, model)

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config)
    trainer: Trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)

    log.info("Starting training!")
    trainer.fit(task, datamodule=datamodule)

    # Check if model saving is enabled
    if config.save_model:
        # Save the model checkpoint
        checkpoint_path = f"saved_models/model_{config.wandb.name}_db={config.data.db}.ckpt"
        trainer.save_checkpoint(checkpoint_path)

        # Optionally, log to WandB
        if config.log_model_to_wandb:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    
    if config.eval_testset:
        log.info("Starting testing!")
        
        #trainer.test(task, datamodule=datamodule)
        trainer.test(ckpt_path="best", datamodule=datamodule)

    wandb.finish()
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


@hydra.main(config_path="config", config_name="train_continue", version_base=None)
@print_exceptions
def continue_training(config: DictConfig):
    set_seed(config)  # Assuming you want to ensure reproducibility

    # Initialize WandB
    wandb.init(project="Informer", entity="space_debri", id="yehwhckr", resume="allow")

    log.info("\n\n\n\n")
    log.info(config.task.name)
    log.info("\n\n\n\n")
    
    # Load the pretrained model
    model_path = config.pretrained_model_path 
    model_architecture = instantiate_model(config.model)
    model = Forecast_Class.load_from_checkpoint(model_path, model=model_architecture, n_classes=4)

    
    
    # log.info(model.scheduler_type)
    # log.info(model.learning_rate)

    # Prepare data
    log.info("Loading data")
    datamodule = instantiate_datamodule(config.data, rng=None)  # Adjust if your datamodule requires a rng
    datamodule.prepare_data()
    datamodule.setup("train")

    # Re-initialize the logger and callbacks, possibly with a new run name or tags to indicate continuation
    logger = WandbLogger()
    log_hyperparameters(logger, config, model)
    callbacks = get_callbacks(config)  # You might want to adjust callbacks for continued training

    # Update trainer to train for additional epochs
    trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)

    # Continue training
    log.info("Continuing training!")
    trainer.fit(model, datamodule=datamodule, ckpt_path=model_path)

    # Save and log the model as before
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

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None

if __name__ == "__main__":
    main()
    #continue_training()
