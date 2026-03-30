import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, d_model, max_len=2081):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (B, L, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """Transformer model."""
    def __init__(self, in_channels, out_channels, hidden_channels=256, num_layers=4, 
                 nhead=8, dim_feedforward=512, dropout=0.1,input_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_channels)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        #self.input_dropout = nn.Dropout(input_dropout)
    
    def forward(self, x):
        # x: (B, C_in, L)
        B, C, L = x.shape
        
        # Transpose to (B, L, C)
        x = x.transpose(1, 2)  # (B, L, C_in)
        
        # Input projection
        x = self.input_proj(x)  # (B, L, hidden_channels)
        #x = self.input_dropout(x)
        
        # Positional encoding
        x = self.pos_encoder(x)  # (B, L, hidden_channels)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, L, hidden_channels)
        
        # Output projection
        x = self.output_proj(x)  # (B, L, out_channels)
        
        # Transpose back to (B, C_out, L)
        x = x.transpose(1, 2)  # (B, out_channels, L)
        
        return x

class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise separable convolution block (identical to the original implementation).
    
    Input / Output: (B, C, L)
    """
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        # Depthwise convolution (per-channel)
        self.dw = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=True,  # Key: allow non-zero output for zero input
        )
        self.gn1 = nn.GroupNorm(1, channels)  # Stable normalization
        self.act = nn.GELU()  # Use GELU instead of ReLU
        
        # Pointwise convolution (channel mixing)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.gn2 = nn.GroupNorm(1, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x  # Save input for residual connection
        
        y = self.dw(x.contiguous())
        y = self.gn1(y)
        y = self.act(y)
        y = self.pw(y)
        y = self.gn2(y)
        y = self.drop(y)
        
        y = y + residual  # Residual connection
        y = self.act(y)
        return y



class ConvModel(nn.Module):
    """
    Convolutional model (identical to the original ChannelARConv).
    Input:  (B, C_in, L)
    Output:  (B, C_out, L)
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        hidden_channels: int = 256,
        num_layers: int = 6,
        kernel_size: int = 7, 
        dropout: float = 0.1, 
        use_dilation: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input projection (with bias)
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=True)

        # Stacked depthwise separable convolution blocks
        blocks = []
        for i in range(num_layers):
            dilation = (2 ** i) if use_dilation else 1
            blocks.append(DepthwiseSeparableBlock(
                channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
        self.blocks = nn.Sequential(*blocks)

        # Output projection (with bias)
        self.output_proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B, C_in, L)
        x = x.contiguous()
        h = self.input_proj(x)     # (B, hidden, L)
        h = self.blocks(h)         # (B, hidden, L) - after multiple residual blocks
        y = self.output_proj(h)    # (B, out, L)
        return y


class MLPModel(nn.Module):
    """MLP model (independent processing at each time step)."""
    def __init__(self, in_channels, out_channels, hidden_channels=256, num_layers=6,
                 dropout=0.1, expansion=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        layers = []
        # Input layer
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels * expansion))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_channels * expansion, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_channels, out_channels))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, C_in, L)
        B, C, L = x.shape
        
        # Transpose to (B, L, C_in)
        x = x.transpose(1, 2)
        
        # Apply MLP per time step
        x = self.mlp(x)  # (B, L, out_channels)
        
        # Transpose back to (B, out_channels, L)
        x = x.transpose(1, 2)
        
        return x


def get_model(config, in_channels, out_channels):
    """
    Build model according to configuration.
    
    Args:
        config: Full configuration dictionary
        in_channels: Number of input channels
        out_channels: Number of output channels
    
    Returns:
        Instantiated model
    """
    model_config = config['model']
    model_name = model_config['name']
    
    # Extract model parameters (exclude 'name')
    params = {k: v for k, v in model_config.items() if k != 'name'}
    
    if model_name == "Transformer":
        return TransformerModel(in_channels, out_channels, **params)
    elif model_name == "Conv":
        return ConvModel(in_channels, out_channels, **params)
    elif model_name == "MLP":
        return MLPModel(in_channels, out_channels, **params)
    else:
        raise ValueError(f"Unknown model: {model_name}")