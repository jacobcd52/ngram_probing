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
        dtype: str = "float32"
    ):
        super().__init__()
        self.d_model = d_model
        self.ngrams = ngrams
        self.device = device
        self.probe_type = probe_type
        self.dtype = getattr(torch, dtype)
        
        # Linear layer for prediction
        self.linear = nn.Linear(d_model, len(ngrams))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the probe."""
        return self.linear(x)
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, positive_weight: float = 1.0) -> torch.Tensor:
        """Compute binary cross-entropy loss with optional positive class weighting."""
        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=torch.tensor([positive_weight], device=self.device, dtype=self.dtype)
        )
    
    def compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute AUROC for each n-gram."""
        probs = torch.sigmoid(logits)
        aurocs = {}
        
        for i, ngram in enumerate(self.ngrams):
            try:
                auroc = roc_auc_score(
                    labels[:, i].cpu().numpy(),
                    probs[:, i].cpu().numpy()
                )
                aurocs[ngram] = auroc
            except ValueError:
                # Handle cases where there's only one class
                aurocs[ngram] = 0.5
                
        return aurocs
    
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