# config.py

from dataclasses import dataclass
from typing import Optional
import os
import torch

@dataclass
class NgramProbingConfig:
    # Model settings
    model_name: str = "EleutherAI/pythia-70m"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    layer: int = -1  # -1 means last layer
    dtype: torch.dtype = torch.bfloat16  # Default to bfloat16 for efficiency
    model_batch_size: int = 2048  # Batch size for model forward passes
    
    # Data settings
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"
    max_texts: int = 200_000  # Default to 200k texts
    ctx_len: int = 128
    batch_size: int = 1024
    chunk_size: int = 50_000  # Number of sequences per chunk for activation caching
    
    # N-gram settings
    ngram_size: int = 3
    top_m_ngrams: int = 1000
    ignore_top_n: int = 100  # Number of most frequent n-grams to ignore
    
    # Training settings
    learning_rate: float = 1e-3
    num_epochs: int = 1
    positive_weight: float = 100
    num_train_tokens: int = int(10e6)
    num_val_tokens: int = int(1e6)
    
    # Output settings
    output_dir: str = "results"
    histogram_bins: int = 20  # Number of bins for histograms
    
    def __post_init__(self):
        # Convert token counts to sequence counts
        self.num_train_sequences = self.num_train_tokens // self.ctx_len
        self.num_val_sequences = self.num_val_tokens // self.ctx_len 