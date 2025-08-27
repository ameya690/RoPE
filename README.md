# RoPE - Rotary Positional Embeddings

This repository contains implementations of different positional encoding methods for transformers, with a focus on Rotary Positional Embeddings (RoPE).

## Contents

- `absolute_positional_encoding.py`: Implementation of absolute positional encoding
- `rotary_positional_encoding.py`: Implementation of rotary positional encoding (RoPE)
- `010_Transformer-3.ipynb`: Jupyter notebook demonstrating the usage of these encodings

## Requirements

- Python 3.6+
- PyTorch
- Jupyter Notebook (for the example notebook)

## Installation

```bash
git clone https://github.com/yourusername/RoPE.git
cd RoPE
pip install -r requirements.txt
```

## Usage

Import and use the positional encoding modules in your project:

```python
from rotary_positional_encoding import RotaryPositionalEncoding
# or
from absolute_positional_encoding import PositionalEncoding
```

## License

MIT
