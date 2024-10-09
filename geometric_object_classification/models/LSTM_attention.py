import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_length, hidden_size]
        attention_weights = self.attention(x).squeeze(2)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Reshape weights to [batch_size, 1, seq_length]
        attention_weights = attention_weights.unsqueeze(1)
        
        # Apply attention weights
        weighted_representation = torch.bmm(attention_weights, x).squeeze(1)
        
        return weighted_representation, attention_weights

class LSTM_Attention_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=False):
        super(LSTM_Attention_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        # Add the attention layer
        self.attention = Attention(hidden_size * (2 if bidirectional else 1))
        
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_features: bool = False):
        # Same initial steps as before...
        device = x.device
        x = x.transpose(1, 2)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Apply attention
        attention_out, attention_weights = self.attention(out)
        
        # Decode the last hidden state
        logits = self.fc(attention_out)
        
        if return_features:
            return logits, attention_out
        else:
            return logits
