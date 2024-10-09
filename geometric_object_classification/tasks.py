import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from torch.optim.lr_scheduler import LambdaLR
import wandb
import umap
import numpy as np
from transformers import get_linear_schedule_with_warmup

class Classification(pl.LightningModule):
    """
    PyTorch Lightning Module for classification tasks. Supports training, validation, and testing
    with various optimizer and learning rate scheduler options. Can optionally log feature
    representations using UMAP and WandB.
    """
    def __init__(self, model, n_classes: int, learning_rate=0.001, weight_decay=1e-5, optimizer_type='adam',
                 scheduler_type='lambda', gradient_clip_val=0.25, feature_plot_2d=False, feature_plot_umap=False):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.feature_plot_2d = feature_plot_2d
        self.feature_plot_umap = feature_plot_umap

        self.loss = nn.CrossEntropyLoss()

        # Metrics for tracking performance
        self.metrics = tm.MetricCollection(
            {
                "top1": tm.Accuracy(task="multiclass", top_k=1, num_classes=n_classes),
                "top3": tm.Accuracy(task="multiclass", top_k=3, num_classes=n_classes),
                "accuracy": tm.Accuracy(task="multiclass", num_classes=n_classes),
                "avg-prec": tm.AveragePrecision(task="multiclass", num_classes=n_classes),
                "precision": tm.Precision(task="multiclass", num_classes=n_classes),
                "recall": tm.Recall(task="multiclass", num_classes=n_classes),
                "f1_macro": tm.F1Score(task="multiclass", num_classes=n_classes, average='macro'),
                "f1_micro": tm.F1Score(task="multiclass", num_classes=n_classes, average='micro')
            }
        )
        self.val_metrics = self.metrics.clone(prefix="val/")
        self.test_metrics = self.metrics.clone(prefix="test/")

        # Logging variables
        self.test_predictions = []
        self.test_ground_truths = []
        self.val_acc_history = []
        self.features = []
        self.data_test = []

    def classify(self, x):
        """
        Classifies input samples by returning the softmax probabilities.
        """
        logits = self.model(x)
        return nn.functional.softmax(logits, dim=-1)

    def training_step(self, batch, batch_idx):
        """
        Performs a training step: computes the loss and logs it.
        """
        x, labels = batch
        logits = self.model(x)
        loss = self.loss(logits, labels)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step: computes the loss and logs validation metrics.
        """
        x, labels = batch
        logits = self.model(x)
        loss = self.loss(logits, labels)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(self.val_metrics(logits, labels), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        """
        Performs a test step: computes the loss, logs test metrics, and optionally stores feature data.
        """
        x, labels = batch
        if self.feature_plot_umap or self.feature_plot_2d:
            logits, feature = self.model(x, return_features=True)
            self.features.append(feature)
            self.data_test.append(x.squeeze(1))
        else:
            logits = self.model(x)

        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.test_predictions.append(preds)
        self.test_ground_truths.append(labels)

        self.log_dict(self.test_metrics(logits, labels), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        """
        Updates the validation accuracy history at the end of each validation epoch.
        """
        avg_val_acc = self.trainer.callback_metrics.get('val/avg_acc')
        if avg_val_acc is not None:
            self.val_acc_history.append(avg_val_acc.item())
            if len(self.val_acc_history) > 5:
                self.val_acc_history.pop(0)

    def on_test_epoch_end(self):
        """
        Logs test results to WandB, including confusion matrix and optional feature plots.
        """
        all_preds = torch.cat(self.test_predictions, dim=0)
        all_labels = torch.cat(self.test_ground_truths, dim=0)

        wandb.log({"conf_mat": wandb.plot.confusion_matrix(
            y_true=all_labels.cpu().numpy(),
            preds=all_preds.cpu().numpy(),
            class_names=["cylinders", "spheres", "circular_plates", "cones"]
        )})

        if self.feature_plot_umap:
            self._log_umap_embeddings(all_labels)

        if self.feature_plot_2d:
            self._log_2d_features(all_labels)

        # Reset lists for the next epoch
        self.test_predictions.clear()
        self.test_ground_truths.clear()
        self.features.clear()
        self.data_test.clear()

    def _log_umap_embeddings(self, all_labels):
        """
        Generates and logs UMAP embeddings to WandB.
        """
        all_features_np = torch.cat(self.features, dim=0).cpu().numpy()
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_embeddings = reducer.fit_transform(all_features_np)

        feature_table = wandb.Table(columns=["x", "y", "label"])
        class_names = ["cylinders", "spheres", "circular_plates", "cones"]
        all_labels_np = all_labels.cpu().numpy()
        for i in range(umap_embeddings.shape[0]):
            feature_table.add_data(umap_embeddings[i, 0], umap_embeddings[i, 1], class_names[all_labels_np[i]])

        wandb.log({"umap_feature_map": feature_table})

    def _log_2d_features(self, all_labels):
        """
        Logs 2D feature scatter plot to WandB.
        """
        all_features_np = torch.cat(self.features, dim=0).cpu().numpy()
        feature_table = wandb.Table(columns=["x", "y", "label"])
        class_names = ["cylinders", "spheres", "circular_plates", "cones"]
        all_labels_np = all_labels.cpu().numpy()
        for i in range(all_features_np.shape[0]):
            feature_table.add_data(all_features_np[i, 0], all_features_np[i, 1], class_names[all_labels_np[i]])

        wandb.log({"features_map": feature_table})

    def configure_optimizers(self):
        """
        Configures optimizers and learning rate schedulers based on specified settings.
        """
        num_training_steps = 5 * 74836

        def lr_scheduler(epoch):
            if len(self.val_acc_history) < 6:
                return 1.0
            elif self.val_acc_history[-1] < torch.mean(torch.tensor(self.val_acc_history[:-1])):
                return 0.7
            return 1.0

        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer, num_training_steps, lr_scheduler)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'gradient_clip_val': self.gradient_clip_val if self.gradient_clip_val != 0.0 else None,
            'gradient_clip_algorithm': 'norm' if self.gradient_clip_val != 0.0 else None
        }

    def _get_optimizer(self):
        """
        Returns the selected optimizer.
        """
        if self.optimizer_type == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def _get_scheduler(self, optimizer, num_training_steps, lr_scheduler):
        """
        Returns the selected learning rate scheduler.
        """
        if self.scheduler_type == 'lambda':
            return LambdaLR(optimizer, lr_lambda=lr_scheduler)
        elif self.scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.scheduler_type == 'linear_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=54, num_training_steps=num_training_steps)
            return {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        elif self.scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif self.scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
            return {'scheduler': scheduler, 'monitor': 'loss/val'}
        elif self.scheduler_type == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
