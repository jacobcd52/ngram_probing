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
    probe_batch_size: int = 128  # Batch size for probe training
    use_random_model: bool = True  # Whether to use random weights instead of pretrained
    
    # Synthetic data settings
    vocab_size: int = 1000  # Size of vocabulary to use (must be <= model's vocab size)
    ctx_len: int = 128  # Context length
    num_contexts: int = 10_000  # Number of contexts to generate per n-gram
    num_ngrams_to_probe: int = 1  # Number of different n-grams to probe
    union_size: int = 10  # Number of n-grams to use in each context
    
    # N-gram settings
    ngram_size: int = 10
    
    # Training settings
    learning_rate: float = 1e-3
    num_epochs: int = 1
    positive_weight: float = 10
    model_batch_size: int = 256  # Batch size for model forward passes
    
    # Output settings
    output_dir: str = os.path.join(os.path.dirname(__file__), "results")  # Put results in package directory
    histogram_bins: int = 20  # Number of bins for histograms
    
    # New validation_split parameter
    validation_split: float = 0.1  # Ratio of data to use for validation 