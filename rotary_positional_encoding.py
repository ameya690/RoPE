import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


def get_rotary_matrix(context_len: int, d_model: int) -> np.ndarray:
    """
    Generate the Rotary Matrix for ROPE (Rotary Position Embedding)
    
    Args:
        context_len (int): Maximum context length
        d_model (int): Dimension of the model embeddings (must be even)
        
    Returns:
        np.ndarray: The rotary matrix of dimension (context_len, d_model, d_model)
    """
    assert d_model % 2 == 0, "d_model must be even for rotary positional encoding"
    
    R = np.zeros((context_len, d_model, d_model))
    positions = np.arange(1, context_len + 1)[:, np.newaxis]
    
    # Create the rotation angles
    slice_i = np.arange(d_model // 2)
    theta = 10000 ** (-2 * (slice_i // 2) / d_model)
    m_theta = positions * theta.reshape(1, -1)
    
    # Create sin and cos values
    cos_values = np.cos(m_theta)
    sin_values = np.sin(m_theta)
    
    # Populate the rotary matrix R using advanced indexing
    R[:, 2*slice_i, 2*slice_i] = cos_values
    R[:, 2*slice_i, 2*slice_i+1] = -sin_values
    R[:, 2*slice_i+1, 2*slice_i] = sin_values
    R[:, 2*slice_i+1, 2*slice_i+1] = cos_values
    
    return R


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) module for PyTorch.
    
    This implementation follows the RoPE (Rotary Position Embedding) approach
    which applies rotation matrices to the embeddings based on their positions.
    """
    def __init__(self, d_model: int, max_len: int = 2048, base: float = 10000.0):
        """
        Initialize the RoPE module.
        
        Args:
            d_model: Dimension of the model embeddings (must be even)
            max_len: Maximum sequence length to precompute rotations for
            base: Base value for frequency calculation
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for rotary positional encoding"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Precompute rotation matrices
        self.register_buffer('R', torch.tensor(
            get_rotary_matrix(max_len, d_model), 
            dtype=torch.float32
        ))
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply rotary positional encoding to the input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Optional tensor of shape (batch_size, seq_len) with position indices.
                     If None, uses positions [0, 1, ..., seq_len-1] for each sequence.
                     
        Returns:
            Tensor with rotary positional encodings applied, same shape as input
        """
        batch_size, seq_len, _ = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).expand(batch_size, -1)
        
        # Ensure positions are within bounds
        positions = positions.clamp(0, self.max_len - 1)
        
        # Get rotation matrices for the given positions
        R = self.R[positions]  # shape: (batch_size, seq_len, d_model, d_model)
        
        # Apply rotation: x @ R (batch matrix multiplication)
        # Reshape x for batch matrix multiplication: (batch_size * seq_len, 1, d_model)
        x_flat = x.reshape(-1, 1, self.d_model)
        # Reshape R for batch matrix multiplication: (batch_size * seq_len, d_model, d_model)
        R_flat = R.reshape(-1, self.d_model, self.d_model)
        # Apply rotation: (batch_size * seq_len, 1, d_model) @ (batch_size * seq_len, d_model, d_model)
        # -> (batch_size * seq_len, 1, d_model)
        x_rotated = torch.bmm(x_flat, R_perm).squeeze(1)
        
        # Reshape back to original shape
        return x_rotated.reshape(batch_size, seq_len, self.d_model)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    positions: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional encoding to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        positions: Optional position indices of shape (batch_size, seq_len)
        
    Returns:
        Tuple of (q_rotated, k_rotated) with rotary positional encodings applied
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Create rotary positional encoding
    rope = RotaryPositionalEncoding(head_dim, max_len=seq_len)
    
    # Reshape for rotary encoding: (batch_size * num_heads, seq_len, head_dim)
    q_reshaped = q.permute(0, 2, 1, 3).reshape(-1, seq_len, head_dim)
    k_reshaped = k.permute(0, 2, 1, 3).reshape(-1, seq_len, head_dim)
    
    # Apply rotary encoding
    q_rotated = rope(q_reshaped, positions)
    k_rotated = rope(k_reshaped, positions)
    
    # Reshape back to original shape
    q_rotated = q_rotated.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    k_rotated = k_rotated.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    
    return q_rotated, k_rotated


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 10
    d_model = 128
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize rotary positional encoding
    rope = RotaryPositionalEncoding(d_model, max_len=seq_len)
    
    # Apply rotary positional encoding
    x_rotated = rope(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_rotated.shape}")
    
    # Example with multi-head attention
    num_heads = 4
    head_dim = d_model // num_heads
    
    # Create random query and key tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Apply rotary positional encoding to query and key
    q_rot, k_rot = apply_rotary_pos_emb(q, k)
    
    print(f"Original query shape: {q.shape}")
    print(f"Rotated query shape: {q_rot.shape}")
    print(f"Original key shape: {k.shape}")
    print(f"Rotated key shape: {k_rot.shape}")
