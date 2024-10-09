import torch
import torch.nn as nn
from transformers import BertModel

class RadarBERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, sequence_length, embedding_option='dense'):
        super(RadarBERTClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Options for embedding layer
        if embedding_option == 'dense':
            self.embedding = nn.Linear(1, self.bert.config.hidden_size)
        elif embedding_option == 'conv1d':
            self.embedding = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Conv1d(in_channels=256, out_channels=self.bert.config.hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(self.bert.config.hidden_size),
                nn.SiLU()
            )
        else:
            raise ValueError('Invalid embedding option')
        
        self.sequence_length = sequence_length
        
        # Classifier head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, x, return_features: bool = False):
        # x shape: (batch_size, channel_in, sequence_length)
        
        # Convert time series data to embeddings
        if isinstance(self.embedding, nn.Linear):
            # Linear embedding, reshape is needed to match dimensions
            x = x.view(-1, self.sequence_length)  # (batch_size, sequence_length)
            x = self.embedding(x)  # (batch_size, sequence_length, hidden_size)
        elif isinstance(self.embedding, nn.Sequential):
            # Conv1D embedding, add an extra dimension and then permute
            x = self.embedding(x)  # (batch_size, hidden_size, sequence_length)
            x = x.permute(0, 2, 1)  # (batch_size, sequence_length, hidden_size)
        
        # Pass through BERT
        outputs = self.bert(inputs_embeds=x)
        
       # Pool the outputs to a single vector per sample
        pooled_output = outputs.pooler_output
        
        # The hidden states of the last layer can be considered as features
        features = outputs.last_hidden_state
        
        # Classifier
        logits = self.classifier(pooled_output)
        
        if return_features:
            return logits, features
        else:
            return logits