#src/ngram_probing.py

import warnings
# Suppress torchvision warning about failed image extension load
warnings.filterwarnings('ignore', category=UserWarning, message='.*Failed to load image Python extension.*')

import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer, utils
from typing import List, Tuple, Dict, Optional
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import NgramProbingConfig
from probe_model import NgramProbe
import pickle
from sklearn.metrics import roc_auc_score
import math
import random

def get_pythia_config():
    """Load or create and save the Pythia config."""
    config_path = "pythia_config.pkl"
    
    if os.path.exists(config_path):
        print("Loading saved Pythia config...")
        with open(config_path, 'rb') as f:
            return pickle.load(f)
    
    print("Loading Pythia model to get config...")
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-70m",
        device="cpu",
        fold_ln=False
    )
    config = model.cfg
    
    print("Saving Pythia config...")
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    return config

class SyntheticDataGenerator:
    def __init__(self, config: NgramProbingConfig):
        self.config = config
        
        # Get the Pythia config
        pythia_config = get_pythia_config()
        
        if config.use_random_model:
            # Initialize the model with random weights
            self.model = HookedTransformer(pythia_config)
            # Initialize the weights (they're set to empty by default)
            self.model.init_weights()
            # Move to correct device
            self.model.to(config.device)
        else:
            # Load the pretrained model
            self.model = HookedTransformer.from_pretrained(
                config.model_name,
                device=config.device,
                fold_ln=False
            )
        
        # Verify vocab size is valid
        assert config.vocab_size <= self.model.cfg.d_vocab, \
            f"vocab_size ({config.vocab_size}) must be <= model's vocab size ({self.model.cfg.d_vocab})"
        
        # Verify ngram size is valid
        assert config.ngram_size <= config.ctx_len, \
            f"ngram_size ({config.ngram_size}) must be <= ctx_len ({config.ctx_len})"
    
    def generate_random_ngram(self) -> Tuple[int, ...]:
        """Generate a random n-gram from the vocabulary."""
        return tuple(random.randint(0, self.config.vocab_size - 1) 
                   for _ in range(self.config.ngram_size))
    
    def generate_context(self, ngram: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a context containing the n-gram at a random position."""
        while True:
            # Generate random position for n-gram
            pos = random.randint(0, self.config.ctx_len - self.config.ngram_size)
            
            # Generate random tokens for the rest of the context
            context = torch.zeros(self.config.ctx_len, dtype=torch.long)
            
            # Fill in random tokens before n-gram
            for i in range(pos):
                context[i] = random.randint(0, self.config.vocab_size - 1)
            
            # Fill in n-gram
            for i, token in enumerate(ngram):
                context[pos + i] = token
            
            # Fill in random tokens after n-gram
            for i in range(pos + self.config.ngram_size, self.config.ctx_len):
                context[i] = random.randint(0, self.config.vocab_size - 1)
            
            # Check if n-gram appears anywhere else in the context
            for i in range(self.config.ctx_len - self.config.ngram_size + 1):
                if i != pos:  # Skip the position where we placed the n-gram
                    if tuple(context[i:i + self.config.ngram_size].tolist()) == ngram:
                        break
            else:  # No other occurrences found
                # Create labels (1 for final token of n-gram, 0 otherwise)
                labels = torch.zeros(self.config.ctx_len, dtype=torch.float)
                labels[pos + self.config.ngram_size - 1] = 1
                return context, labels
    
    def generate_dataset(self, ngrams: List[Tuple[int, ...]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a dataset containing one of the specified n-grams in each context."""
        # Generate random tokens for all contexts at once
        tokens = torch.randint(0, self.config.vocab_size, 
                             (self.config.num_contexts, self.config.ctx_len), 
                             device=self.config.device)
        
        # Generate random positions for all contexts at once
        max_pos = self.config.ctx_len - len(ngrams[0]) + 1
        positions = torch.randint(0, max_pos, (self.config.num_contexts,), device=self.config.device)
        
        # Generate random n-gram indices for each context
        ngram_indices = torch.randint(0, len(ngrams), (self.config.num_contexts,), device=self.config.device)
        
        # Convert n-grams to tensor for easier indexing
        ngrams_tensor = torch.tensor(ngrams, device=self.config.device)
        
        # Place n-grams at their positions
        for i in range(len(ngrams[0])):
            tokens[torch.arange(self.config.num_contexts, device=self.config.device), 
                  positions + i] = ngrams_tensor[ngram_indices, i]
        
        # Create labels (1 for positions where n-grams end)
        labels = torch.zeros((self.config.num_contexts, self.config.ctx_len), 
                           dtype=torch.float32, device=self.config.device)
        labels[torch.arange(self.config.num_contexts, device=self.config.device), 
               positions + len(ngrams[0]) - 1] = 1.0
        
        return tokens, labels
    
    def get_activations(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get activations for a batch of tokens."""
        tokens = tokens.to(self.config.device)
        all_activations = []
        
        # Process in smaller batches with progress bar
        pbar = tqdm(range(0, len(tokens), self.config.model_batch_size), 
                   desc="Getting model activations")
        
        for i in pbar:
            batch_end = min(i + self.config.model_batch_size, len(tokens))
            batch_tokens = tokens[i:batch_end]
            
            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    batch_tokens,
                    return_type="loss",
                    names_filter=lambda name: f"blocks.{self.config.layer}.hook_resid_post"
                )
            
            if self.config.layer < 0:
                layer = self.model.cfg.n_layers + self.config.layer
            batch_activations = cache[f"blocks.{layer}.hook_resid_post"]
            
            all_activations.append(batch_activations)
            
            # Clear cache after each batch
            del cache
            torch.cuda.empty_cache()
        
        # Concatenate all batches
        activations = torch.cat(all_activations, dim=0)
        return activations.to(dtype=self.config.dtype)

class NgramProber:
    def __init__(self, config: NgramProbingConfig):
        self.config = config
        self.data_generator = SyntheticDataGenerator(config)
        
        # Create necessary directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "plots"), exist_ok=True)
    
    def train_and_evaluate_probe(
        self,
        ngrams: List[Tuple[int, ...]],
        probe_type: str
    ) -> float:
        """Train and evaluate a probe for a set of n-grams and position type."""
        # Generate dataset
        tokens, labels = self.data_generator.generate_dataset(ngrams)
        labels = labels.to(self.config.device)  # Move labels to correct device
        
        # Debug prints for labels
        print(f"\nLabels tensor shape: {labels.shape}")
        print(f"Number of positive labels: {labels.sum().item()}")
        print(f"Number of negative labels: {(labels == 0).sum().item()}")
        print(f"Unique values in labels: {torch.unique(labels).tolist()}")
        
        # Get activations
        activations = self.data_generator.get_activations(tokens)
        
        # Split into train and validation
        num_train = int(len(tokens) * (1 - self.config.validation_split))
        print(f"\nDataset split:")
        print(f"Total contexts: {len(tokens)}")
        print(f"Training contexts: {num_train}")
        print(f"Validation contexts: {len(tokens) - num_train}")
        
        train_activations = activations[:num_train]
        train_labels = labels[:num_train]
        val_activations = activations[num_train:]
        val_labels = labels[num_train:]
        
        # Debug prints for validation labels
        print(f"\nValidation labels shape: {val_labels.shape}")
        print(f"Number of positive labels in validation: {val_labels.sum().item()}")
        print(f"Number of negative labels in validation: {(val_labels == 0).sum().item()}")
        print(f"Unique values in validation labels: {torch.unique(val_labels).tolist()}")
        
        # Create probe
        probe = NgramProbe(
            d_model=self.data_generator.model.cfg.d_model,
            ngrams=ngrams,  # Pass all n-grams
            device=self.config.device,
            probe_type=probe_type,
            dtype=self.config.dtype
        ).to(self.config.device)
        
        # Training loop
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.config.learning_rate)
        best_val_auroc = 0.0
        best_probe_state = probe.state_dict()  # Initialize with initial state
        
        for epoch in range(self.config.num_epochs):
            # Training
            probe.train()
            total_loss = 0
            num_batches = 0
            
            # Create progress bar for training
            pbar = tqdm(range(0, len(train_activations), self.config.probe_batch_size), 
                       desc=f"Training batches")
            
            for i in pbar:
                batch_end = min(i + self.config.probe_batch_size, len(train_activations))
                batch_activations = train_activations[i:batch_end]
                batch_labels = train_labels[i:batch_end]
                
                logits = probe(batch_activations).squeeze(-1)  # Squeeze the last dimension
                loss = probe.compute_loss(logits, batch_labels, self.config.positive_weight)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear memory after each batch
                del logits, loss
                torch.cuda.empty_cache()
            
            # Validation
            probe.eval()
            with torch.no_grad():
                val_logits = probe(val_activations).squeeze(-1)
                val_auroc = probe.compute_metrics(val_logits, val_labels)[ngrams[0]]  # Use first n-gram for validation
                
                # Update best validation AUROC
                if val_auroc > best_val_auroc:
                    best_val_auroc = val_auroc
                    # Save best model
                    best_probe_state = probe.state_dict()
        
        # Load best model for final evaluation
        probe.load_state_dict(best_probe_state)
        probe.eval()
        with torch.no_grad():
            logits = probe(activations).squeeze(-1)
            auroc = probe.compute_metrics(logits, labels)[ngrams[0]]  # Use first n-gram for final evaluation
        
        return auroc
    
    def run_probing(self):
        """Run probing experiments on multiple random n-grams."""
        results = {
            'first': [],
            'final': []
        }
        
        for i in tqdm(range(self.config.num_ngrams_to_probe), desc="Probing n-grams"):
            # Generate union_size many n-grams
            ngrams = [self.data_generator.generate_random_ngram() for _ in range(self.config.union_size)]
            
            # Train and evaluate both probe types
            for probe_type in ['first', 'final']:
                auroc = self.train_and_evaluate_probe(ngrams, probe_type)
                results[probe_type].append({
                    'ngrams': ngrams,  # Store all n-grams
                    'auroc': auroc
                })
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, List[Dict]]):
        """Save results and create plots."""
        # Save raw results
        output_file = os.path.join(self.config.output_dir, "probe_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        for probe_type in ['first', 'final']:
            aurocs = [r['auroc'] for r in results[probe_type]]
            
            # Histogram of AUROC scores
            plt.figure(figsize=(10, 6))
            plt.hist(aurocs, bins=self.config.histogram_bins)
            plt.title(f"Distribution of AUROC Scores ({probe_type}-position probe)")
            plt.xlabel("AUROC Score")
            plt.ylabel("Count")
            plt.savefig(os.path.join(self.config.output_dir, "plots", f"auroc_distribution_{probe_type}.png"))
            plt.close()
        
        # Print summary statistics
        print("\nResults Summary:")
        for probe_type in ['first', 'final']:
            aurocs = [r['auroc'] for r in results[probe_type]]
            print(f"\n{probe_type.capitalize()}-position probe:")
            print(f"Mean AUROC: {np.mean(aurocs):.3f}")
            print(f"Std AUROC: {np.std(aurocs):.3f}")
            print(f"Min AUROC: {np.min(aurocs):.3f}")
            print(f"Max AUROC: {np.max(aurocs):.3f}")

def main():
    config = NgramProbingConfig()
    prober = NgramProber(config)
    results = prober.run_probing()

if __name__ == "__main__":
    main() 