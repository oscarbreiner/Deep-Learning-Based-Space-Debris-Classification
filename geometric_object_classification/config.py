from pathlib import Path
from omegaconf import DictConfig
import wandb

from .data import RadarDataModule
from .models.residual_network import ResidualNetwork
from .models.MLP import MLP
from .models.bert import RadarBERTClassifier
from .models.Informer2020.model import Informer_Classification, Informer_Forecaster
from .models.LSTM import LSTMClassifier
from .models.LSTM_attention import LSTM_Attention_Classifier
from .models.transformer import TimeSeriesTransformer
from .tasks import Classification
from .task_forecasting import Forecast_Class

def instantiate_datamodule(config: DictConfig, rng=None):
    if config.name == "radar_echo_return_simple_shapes":
        return RadarDataModule(Path(config.root), stats_file="stats.json", batch_size=config.batch_size, max_samples_per_class=config.max_samples_per_class, db=config.db)

def instantiate_model(config: DictConfig):
    if config.name == "residual_network":
        print("deploying mode: residual_network")
        
        return ResidualNetwork(res_layers_per_block=config.res_layers_per_block, 
                               num_classes=config.num_classes, 
                               activation=config.activation, 
                               use_noise=config.use_noise,
                               noise_stddev=config.noise_stddev,
                               apply_he=config.apply_he,
                               fc_units=config.fc_units,
                               in_out_channels=config.in_out_channels)
    
    elif config.name == "MLP":
        print("deploying mode: MLP")
        return MLP(input_size=config.input_size,
                    num_classes=config.num_classes,
                    layer_sizes=config.layer_sizes,
                    dropout_rate=config.dropout_rate
                    )
    
    elif config.name == "lstm_attention":
        print("deploying mode: lstm_attention")
        
        return LSTM_Attention_Classifier(input_size=config.input_size,
                              num_classes=config.num_classes,
                              num_layers=config.num_layers,
                              hidden_size=config.hidden_size,
                              bidirectional=config.bidirectional
                              )
    
    elif config.name == "bert":
        print("deploying mode: bert")
        
        return RadarBERTClassifier(
            bert_model_name=config.bert_model_name,
            num_classes=config.num_classes,
            sequence_length=config.sequence_length,
            embedding_option=config.embedding)
    
    elif config.name == "lstm":
        print("deploying mode: lstm")
        
        return LSTMClassifier(input_size=config.input_size,
                              num_classes=config.num_classes,
                              num_layers=config.num_layers,
                              hidden_size=config.hidden_size,
                              bidirectional=config.bidirectional
                              )
    
    elif config.name == "transformer":
        print("deploying mode: Transformer")
        return TimeSeriesTransformer(num_features=config.num_features,
                                    sequence_length=config.sequence_length,
                                    num_classes=config.num_classes,
                                    embedding_option=config.embedding_option,
                                    num_layers=config.num_layers,
                                    dim_feedforward=config.dim_feedforward,
                                    nhead=config.nhead,
                                    embed_dim=config.embed_dim)
    else:
        raise ValueError("Model name not recognized")   


def instantiate_task(config: DictConfig, model, datamodule):
    if config.name == "classification":
        return Classification(model, 
                              n_classes=datamodule.n_classes, 
                              weight_decay=config.weight_decay, 
                              learning_rate=config.learning_rate,
                              optimizer_type=config.optimizer_type,
                              scheduler_type=config.scheduler_type,
                              gradient_clip_val=config.gradient_clip_val,
                              feature_plot_2d=config.feature_plot_2d,
                              feature_plot_umap=config.feature_plot_umap)
    elif config.name == "forecast_class":
        return Forecast_Class(model, 
                              output_size=config.output_size, 
                              weight_decay=config.weight_decay, 
                              learning_rate=config.learning_rate,
                              optimizer_type=config.optimizer_type,
                              scheduler_type=config.scheduler_type,
                              gradient_clip_val=config.gradient_clip_val,
                              feature_plot_2d=config.feature_plot_2d,
                              feature_plot_umap=config.feature_plot_umap,
                              classification=config.classification)
