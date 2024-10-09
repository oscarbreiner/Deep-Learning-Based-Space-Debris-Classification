import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for time series classification.
    Supports embedding options and includes a positional encoding mechanism.
    """
    def __init__(self, num_features, num_classes, sequence_length, embedding_option='conv1d', 
                 num_layers=1, nhead=1, dim_feedforward=2048, embed_dim=256):
        super(TimeSeriesTransformer, self).__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

        # Embedding layer based on the selected option
        if embedding_option == 'dense':
            self.embedding = nn.Linear(num_features, num_features)
        elif embedding_option == 'conv1d':
            self.embedding = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Conv1d(in_channels=256, out_channels=self.embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(self.embed_dim),
                nn.SiLU()
            )
        else:
            raise ValueError('Invalid embedding option')

        # Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=self.embed_dim, max_len=sequence_length)

        # Classification Head
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Convert input to embeddings
        if isinstance(self.embedding, nn.Linear):
            x = x.permute(0, 2, 1)  # (batch_size, sequence_length, num_features)
            x = self.embedding(x)
        elif isinstance(self.embedding, nn.Sequential):
            x = self.embedding(x)  # (batch_size, embed_dim, sequence_length)
            x = x.permute(0, 2, 1)  # (batch_size, sequence_length, embed_dim)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)

        # Pooling to obtain a fixed-size output for classification
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)
        return logits

class PositionalEncoding(nn.Module):
    """
    Positional encoding module for adding positional information to the input features.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Positional encoded tensor.
        """
        return x + self.pe[:x.size(0), :]
