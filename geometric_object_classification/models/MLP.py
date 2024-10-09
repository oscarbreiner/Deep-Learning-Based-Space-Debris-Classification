import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for classification tasks.
    """
    def __init__(self, input_size=501, num_classes=10, layer_sizes=[256, 128], dropout_rate=0.5):
        super(MLP, self).__init__()

        # Create layers for feature extraction
        layers = []
        for i, layer_size in enumerate(layer_sizes):
            in_features = input_size if i == 0 else layer_sizes[i - 1]
            layers.append(nn.Linear(in_features, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        self.feature_extraction = nn.Sequential(*layers)
        self.classifier = nn.Linear(layer_sizes[-1], num_classes)

    def forward(self, x, return_features: bool = False):
        """
        Forward pass through the MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            return_features (bool): Whether to return the features along with the logits.

        Returns:
            torch.Tensor: Logits if return_features is False, otherwise (logits, features).
        """
        x = x.view(x.size(0), -1)  # Flatten input except for the batch dimension
        features = self.feature_extraction(x)
        logits = self.classifier(features)

        return (logits, features) if return_features else logits
