import torch
import torch.nn as nn

def nan_hook(module, input, output):
    """
    Hook function to detect NaN values in a module's output.
    """
    if torch.isnan(output).any():
        raise RuntimeError(f"NaN detected in {module.__class__.__name__} with input {input}")

def xavier_init(m):
    """
    Xavier initialization for Conv1d and Linear layers.
    """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def he_init(m):
    """
    He initialization for Conv1d and Linear layers with ReLU activation.
    """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class NoiseLayer(nn.Module):
    """
    Adds Gaussian noise to the input during training.
    """
    def __init__(self, noise_stddev=0.1):
        super(NoiseLayer, self).__init__()
        self.noise_stddev = noise_stddev

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.noise_stddev
        return x

class ResidualBlock(nn.Module):
    """
    Residual Block with optional noise addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 use_noise=False, noise_stddev=0.1, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.use_noise = use_noise
        self.activation_func = nn.SiLU() if activation == 'silu' else nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.noise = NoiseLayer(noise_stddev) if use_noise else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.activation_func(self.bn1(self.conv1(x)))
        out = self.noise(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        return self.activation_func(out)

class ResidualNetwork(nn.Module):
    """
    Residual Network for time series classification with optional noise addition.
    """
    def __init__(self, res_layers_per_block, num_classes=4, activation='relu', use_noise=False, 
                 noise_stddev=0.1, apply_he=False, fc_units=32, in_out_channels=None):
        super(ResidualNetwork, self).__init__()
        self.in_channels = 1
        self.activation = activation
        self.use_noise = use_noise
        self.noise_stddev = noise_stddev
        self.fc_units = fc_units

        # Default channel configuration if not provided
        if in_out_channels is None:
            in_out_channels = [[32, 64], [64, 128], [128, 256], [256, 512]]

        # Pre-processing layers
        self.pre_layers = nn.Sequential(
            nn.Conv1d(self.in_channels, in_out_channels[0][0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(in_out_channels[0][0]),
            nn.SiLU() if activation == 'silu' else nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Ensure in_out_channels and res_layers_per_block have matching lengths
        assert len(in_out_channels) == len(res_layers_per_block), "in_out_channels and res_layers_per_block must match in length"
        
        # Create residual layers
        self.res_layers = nn.ModuleList([
            self._make_res_layer(in_channels, out_channels, kernel_size=3, stride=(2 if i > 0 else 1), num_blocks=res_layers_per_block[i])
            for i, (in_channels, out_channels) in enumerate(in_out_channels)
        ])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_out_channels[-1][-1], fc_units),
            nn.BatchNorm1d(fc_units),
            nn.SiLU() if activation == 'silu' else nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classifier
        self.classifier = nn.Linear(fc_units, num_classes)

        # Apply He initialization if specified
        if apply_he:
            self.apply(he_init)

    def _make_res_layer(self, in_channels, out_channels, kernel_size, stride, num_blocks):
        """
        Constructs a residual layer with the specified number of blocks.
        """
        layers = [ResidualBlock(in_channels, out_channels, kernel_size, stride, padding=1, 
                                use_noise=self.use_noise, noise_stddev=self.noise_stddev, activation=self.activation)]
        layers.extend([
            ResidualBlock(out_channels, out_channels, kernel_size, stride=1, padding=1, 
                          use_noise=self.use_noise, noise_stddev=self.noise_stddev, activation=self.activation)
            for _ in range(1, num_blocks)
        ])
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).
            return_features (bool): Whether to return the feature representation along with the logits.

        Returns:
            torch.Tensor: Logits if return_features is False, otherwise (logits, features).
        """
        out = self.pre_layers(x)
        for layer in self.res_layers:
            out = layer(out)
        
        out = self.avgpool(out).flatten(1)
        features = self.feature_extractor(out)
        logits = self.classifier(features)

        return (logits, features) if return_features else logits
