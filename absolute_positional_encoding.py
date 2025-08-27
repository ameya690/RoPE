import numpy as np
import torch

def sinusoidal_positional_encoding(max_position, d_model):
    """
    Generate sinusoidal positional encodings.
    
    Args:
        max_position (int): Maximum number of positions to generate encodings for
        d_model (int): Dimension of the model embeddings
        
    Returns:
        numpy.ndarray: Positional encodings of shape (max_position, d_model)
    """
    position = np.arange(max_position)[:, np.newaxis]
    # The original formula pos / 10000^(2i/d_model) is equivalent to pos * (1 / 10000^(2i/d_model)).
    # This version is used for numerical stability
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pos_enc = np.zeros((max_position, d_model))
    pos_enc[:, 0::2] = np.sin(position * div_term)  # even indices
    pos_enc[:, 1::2] = np.cos(position * div_term)  # odd indices
    
    return pos_enc

class SinusoidalPositionalEncoding:
    """
    Sinusoidal Positional Encoding module that can be used in PyTorch models.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): Dimension of the model embeddings
            max_len (int): Maximum sequence length to precompute encodings for
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Precompute positional encodings
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * 
                         (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)  # even indices
        pe[:, 1::2] = np.cos(position * div_term)  # odd indices
        
        # Convert to PyTorch tensor and add batch dimension
        self.register_buffer('pe', torch.FloatTensor(pe).unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor of same shape as input with positional encodings added
        """
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1)]


if __name__ == "__main__":
    # Example usage
    d_model = 128
    max_len = 100
    
    # Using the standalone function
    pos_enc = sinusoidal_positional_encoding(max_len, d_model)
    print(f"Positional encoding shape: {pos_enc.shape}")
    
    # Using the PyTorch module
    pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
    x = torch.randn(5, 10, d_model)  # batch_size=5, seq_len=10, d_model=128
    output = pos_encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
