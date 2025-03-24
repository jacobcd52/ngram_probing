# probe_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import roc_auc_score
import os 

class NgramProbe(nn.Module):
    def __init__(
        self,
        d_model: int,
        ngrams: List[Tuple[int, ...]],
        device: str = "cuda",
        probe_type: str = "first",
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.d_model = d_model
        self.ngrams = ngrams
        self.device = device
        self.probe_type = probe_type
        self.dtype = dtype
        
        # Linear layer for prediction
        self.linear = nn.Linear(d_model, 1).to(device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the probe."""
        # x shape: [batch_size, seq_len, d_model]
        # We want to predict if each position is the end of any of our n-grams
        
        # Project to scalar logits
        logits = self.linear(x.to(dtype=self.dtype))  # [batch_size, seq_len, 1]
        
        return logits
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, positive_weight: float = 1.0) -> torch.Tensor:
        """Compute binary cross-entropy loss with optional positive class weighting."""
        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=torch.tensor([positive_weight], device=self.device, dtype=self.dtype)
        )
    
    def compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[Tuple[int, ...], float]:
        """Compute AUROC for each n-gram."""
        metrics = {}
        probs = torch.sigmoid(logits).float()  # Convert to float32
        
        # Get valid positions based on probe type
        if self.probe_type == "first":
            # Only look at positions where n-gram could start
            valid_positions = slice(0, -len(self.ngrams[0]) + 1)
        else:  # "final"
            # Only look at positions where n-gram could end
            valid_positions = slice(len(self.ngrams[0]) - 1, None)
        
        # Get valid positions from labels and probabilities
        y_true = labels[:, valid_positions].flatten().cpu().numpy()
        y_score = probs[:, valid_positions].flatten().cpu().numpy()
        
        # Compute AUROC for valid positions
        metrics[self.ngrams[0]] = roc_auc_score(y_true, y_score)
        
        return metrics
    
    def save(self, path: str):
        """Save the probe model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'd_model': self.d_model,
            'ngrams': self.ngrams,
            'probe_type': self.probe_type,
            'dtype': self.dtype
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cuda") -> 'NgramProbe':
        """Load a saved probe model."""
        checkpoint = torch.load(path)
        probe = cls(
            d_model=checkpoint['d_model'],
            ngrams=checkpoint['ngrams'],
            device=device,
            probe_type=checkpoint['probe_type'],
            dtype=checkpoint['dtype']
        )
        probe.load_state_dict(checkpoint['model_state_dict'])
        return probe 