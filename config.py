# config.py

from dataclasses import dataclass
from typing import Optional
import os
import torch
import math

@dataclass
class NgramProbingConfig:
    # Model settings
    model_name: str = "EleutherAI/pythia-70m"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    layer: int = -1  # -1 means last layer
    dtype: torch.dtype = torch.bfloat16  # Default to bfloat16 for efficiency
    model_batch_size: int = 512  # Batch size for model forward passes
    
    # Data settings
    dataset_name: str = "kh4dien/fineweb-100m-sample"
    dataset_split: str = "train"
    max_texts: int = 200_000  # Default to 200k texts
    ctx_len: int = 128
    batch_size: int = 512
    chunk_size: int = 10_000  # Reduced from 50k to get more chunks
    
    # N-gram settings
    ngram_size: int = 3
    top_m_ngrams: int = 1000
    
    # Training settings
    learning_rate: float = 1e-3
    num_epochs: int = 1
    positive_weight: float = 500
    num_train_tokens: int = int(10e6)
    num_val_tokens: int = int(10e6)
    
    # Output settings
    output_dir: str = os.path.join(os.path.dirname(__file__), "results")  # Put results in package directory
    histogram_bins: int = 20  # Number of bins for histograms
    
    def __post_init__(self):
        # Convert token counts to sequence counts using ceil division
        self.num_train_sequences = math.ceil(self.num_train_tokens / self.ctx_len)
        self.num_val_sequences = math.ceil(self.num_val_tokens / self.ctx_len) 