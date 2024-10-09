import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for time series data.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

        # Fully connected layer for classification
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x, return_features: bool = False):
        """
        Forward pass through the LSTM classifier.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).
            return_features (bool): Whether to return the features along with the logits.

        Returns:
            torch.Tensor: Logits if return_features is False, otherwise (logits, features).
        """
        device = x.device

        # Reshape input to [batch_size, sequence_length, input_size]
        x = x.transpose(1, 2)

        # Initialize hidden and cell states
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size, device=device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Extract the features from the last time step
        features = out[:, -1, :]
        logits = self.fc(features)

        return (logits, features) if return_features else logits
