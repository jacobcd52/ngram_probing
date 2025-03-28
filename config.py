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
    dataset_name: str = "Elriggs/openwebtext-100k"
    dataset_split: str = "train"
    max_texts: int = 10_000
    ctx_len: int = 128
    batch_size: int = 512
    chunk_size: int = 10_000  # Reduced from 50k to get more chunks
    
    # N-gram settings
    ngram_size: int = 1
    top_m_ngrams: int = 100
    union_size: int = 1  # Size of each n-gram set to probe for
    frequency_ratio: float = 1.5  # Maximum ratio between frequencies of n-grams in a set
    
    # Training settings
    learning_rate: float = 1e-3
    num_epochs: int = 1
    positive_weight: float = 100
    num_train_tokens: int = int(5e6)
    num_val_tokens: int = int(5e6)
    
    # Output settings
    output_dir: str = os.path.join(os.path.dirname(__file__), "results")  # Put results in package directory
    histogram_bins: int = 20  # Number of bins for histograms
    
    # New validation_split parameter
    validation_split: float = 0.5  # Ratio of data to use for validation 