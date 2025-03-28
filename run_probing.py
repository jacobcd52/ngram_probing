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
from torcheval.metrics import BinaryAUROC

class NgramProber:
    def __init__(self, config: NgramProbingConfig):
        self.config = config
        self.model = HookedTransformer.from_pretrained(
            config.model_name,
            device=config.device,
            fold_ln=False
        )
        self.ngram_counts = None
        self.tokens = None  # Cache for tokenized dataset
        
        # Create necessary directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "cache"), exist_ok=True)
        
    def get_cache_path(self) -> str:
        """Get the path for caching n-gram counts."""
        cache_name = (f"ngram_counts_"
                     f"size{self.config.ngram_size}_"
                     f"ctx{self.config.ctx_len}_"
                     f"top{self.config.top_m_ngrams}.pkl")
        return os.path.join(self.config.output_dir, "cache", cache_name)
        
    def save_ngram_counts(self):
        """Save n-gram counts to a cache file."""
        if self.ngram_counts is None:
            raise ValueError("No n-gram counts to save!")
            
        cache_path = self.get_cache_path()
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'ngram_counts': self.ngram_counts,
                'config': {
                    'ngram_size': self.config.ngram_size,
                    'ctx_len': self.config.ctx_len,
                    'top_m_ngrams': self.config.top_m_ngrams
                }
            }, f)
        
    def load_ngram_counts(self) -> bool:
        """
        Load n-gram counts from cache file if it exists and matches current config.
        Returns True if successfully loaded, False otherwise.
        """
        cache_path = self.get_cache_path()
        if not os.path.exists(cache_path):
            return False
            
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                
            # Verify configuration matches
            cached_config = data['config']
            if (cached_config['ngram_size'] != self.config.ngram_size or
                cached_config['ctx_len'] != self.config.ctx_len or
                cached_config['top_m_ngrams'] != self.config.top_m_ngrams):
                return False
                
            self.ngram_counts = data['ngram_counts']
            
            return True
            
        except Exception as e:
            print(f"Error loading n-grams: {str(e)}")
            return False
        
    def load_dataset(self):
        """Load the dataset."""
        print(f"\nLoading first {self.config.max_texts} texts from {self.config.dataset_name}")
        return load_dataset(
            self.config.dataset_name,
            split=f"{self.config.dataset_split}[:{self.config.max_texts}]"
        )
        
    def get_tokens(self, dataset, force_recompute=False):
        """Get or load cached tokens."""
        cache_dir = os.path.join(self.config.output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_path = os.path.join(cache_dir, "cached_tokens")
        
        # Check if all chunks exist and are valid
        if not force_recompute:
            chunk_idx = 0
            all_chunks_exist = True
            while True:
                chunk_path = f"{base_path}_chunk{chunk_idx}.pt"
                if not os.path.exists(chunk_path):
                    break
                chunk_idx += 1
            if chunk_idx > 0:
                # Verify all chunks are valid
                for i in range(chunk_idx):
                    chunk = self.load_tokens_chunk(i)
                    if chunk is None:
                        all_chunks_exist = False
                        break
                if all_chunks_exist:
                    print(f"Found {chunk_idx} valid token chunks")
                    return None  # Return None to indicate we're using chunked loading
        
        # Tokenize in chunks
        chunk_size = 10000  # Process 10k texts at a time
        num_chunks = (len(dataset) + chunk_size - 1) // chunk_size
        
        print(f"\nTokenizing {len(dataset)} texts in {num_chunks} chunks...")
        for chunk_idx in tqdm(range(num_chunks)):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(dataset))
            
            chunk_dataset = dataset.select(range(start_idx, end_idx))
            chunk_tokens = utils.tokenize_and_concatenate(
                chunk_dataset,
                self.model.tokenizer,
                max_length=self.config.ctx_len,
                add_bos_token=False
            )['tokens']
            
            # Save this chunk
            chunk_path = f"{base_path}_chunk{chunk_idx}.pt"
            torch.save(chunk_tokens, chunk_path)
            
            # Clear memory
            del chunk_tokens
            torch.cuda.empty_cache()
        
        print(f"Saved {num_chunks} token chunks")
        return None  # Return None to indicate we're using chunked loading
        
    def get_ngrams(self, tokens: Optional[torch.Tensor] = None) -> Dict[str, int]:
        """Extract n-grams from tokenized texts."""
        if self.load_ngram_counts():
            return self.ngram_counts
        
        # Count total tokens and positions across all chunks
        total_tokens = 0
        total_positions = 0
        chunk_idx = 0
        
        # First pass: count total tokens and positions
        while True:
            chunk = self.load_tokens_chunk(chunk_idx)
            if chunk is None:
                break
            total_tokens += len(chunk)
            total_positions += len(chunk) * self.config.ctx_len - (self.config.ngram_size - 1) * len(chunk)
            chunk_idx += 1
        
        if chunk_idx == 0:
            raise ValueError("No token chunks found! Run get_tokens() first.")
        
        total_positions = total_tokens * self.config.ctx_len - (self.config.ngram_size - 1) * total_tokens
        
        # Use running counter with frequency-based filtering
        ngram_counts = Counter()
        min_count = int(1e-5 * total_positions)  # Minimum count needed to meet frequency threshold
        max_count = int(1e-2 * total_positions)  # Maximum count allowed by frequency threshold
        
        print(f"\nCounting n-grams (filtering counts between {min_count} and {max_count})...")
        
        # Second pass: count n-grams
        for chunk_idx in tqdm(range(chunk_idx), desc="Processing chunks"):
            chunk = self.load_tokens_chunk(chunk_idx)
            if chunk is None:
                continue
            
            for batch_start in range(0, len(chunk), self.config.model_batch_size):
                batch_end = min(batch_start + self.config.model_batch_size, len(chunk))
                batch = chunk[batch_start:batch_end].cpu().numpy()
                
                for seq in batch:
                    # Only count n-grams that could potentially meet our frequency thresholds
                    seq_ngrams = [
                        tuple(seq[j:j + self.config.ngram_size].tolist())
                        for j in range(len(seq) - self.config.ngram_size + 1)
                    ]
                    
                    # Update counter with filtering
                    for ngram in seq_ngrams:
                        current_count = ngram_counts[ngram]
                        if current_count < max_count:  # Only count if below max threshold
                            ngram_counts[ngram] = current_count + 1
                
                del batch
                if batch_start % (5 * self.config.model_batch_size) == 0:
                    torch.cuda.empty_cache()
            
            del chunk
            torch.cuda.empty_cache()
        
        print(f"\nDebug: Found {len(ngram_counts)} unique n-grams")
        print(f"First few n-grams and their counts: {list(ngram_counts.most_common())[:5]}")
        
        # Filter n-grams by frequency thresholds
        selected_ngrams = []
        for ngram, count in ngram_counts.most_common():
            if min_count <= count <= max_count:
                selected_ngrams.append((ngram, count))
                if len(selected_ngrams) >= self.config.top_m_ngrams:
                    break
        
        print(f"\nDebug: Selected {len(selected_ngrams)} n-grams after frequency filtering")
        print(f"First few selected n-grams: {selected_ngrams[:5]}")
        
        # Store selected n-grams
        self.ngram_counts = dict(selected_ngrams)
        
        # Print information about selected n-grams
        print(f"\nSelected {len(selected_ngrams)} n-grams:")
        frequencies = [count / total_positions for ngram, count in selected_ngrams]
        min_freq = frequencies[-1]
        max_freq = frequencies[0]
        print(f"Frequency range: {min_freq:.2e} to {max_freq:.2e}")
        
        # If union_size > 1, create sets of n-grams with similar frequencies
        if self.config.union_size > 1:
            print(f"\nCreating sets of {self.config.union_size} n-grams with similar frequencies...")
            ngram_sets = []
            used_ngrams = set()
            
            # Sort n-grams by frequency for easier processing
            sorted_ngrams = sorted(selected_ngrams, key=lambda x: x[1], reverse=True)
            
            for i, (base_ngram, base_count) in enumerate(sorted_ngrams):
                if base_ngram in used_ngrams:
                    continue
                    
                # Create a set starting with this n-gram
                current_set = [base_ngram]
                used_ngrams.add(base_ngram)
                
                # Look for other n-grams with similar frequencies
                for j, (candidate_ngram, candidate_count) in enumerate(sorted_ngrams[i+1:], start=i+1):
                    if candidate_ngram in used_ngrams:
                        continue
                        
                    # Check if frequencies are within the allowed ratio
                    if (base_count / candidate_count <= self.config.frequency_ratio and 
                        candidate_count / base_count <= self.config.frequency_ratio):
                        current_set.append(candidate_ngram)
                        used_ngrams.add(candidate_ngram)
                        
                        if len(current_set) == self.config.union_size:
                            break
                
                if len(current_set) == self.config.union_size:
                    ngram_sets.append(tuple(current_set))  # Convert list to tuple
            
            print(f"Created {len(ngram_sets)} sets of {self.config.union_size} n-grams")
            print(f"First few sets: {ngram_sets[:5]}")
            
            # Update ngram_counts to use sets as keys
            new_ngram_counts = {}
            for ngram_set in ngram_sets:
                # Use the tuple of n-grams as the key
                new_ngram_counts[ngram_set] = sum(self.ngram_counts[ngram] for ngram in ngram_set)
            
            self.ngram_counts = new_ngram_counts
            print(f"\nDebug: Final n-gram sets and their counts: {list(self.ngram_counts.items())[:5]}")
            print(f"Debug: Type of first n-gram set: {type(list(self.ngram_counts.keys())[0])}")
            print(f"Debug: Length of first n-gram set: {len(list(self.ngram_counts.keys())[0])}")
        
        self.save_ngram_counts()
        return self.ngram_counts
    
    def get_activations_batch(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get activations for a batch of tokens."""
        tokens = tokens.to(self.config.device)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens,
                return_type="loss",
                names_filter=lambda name: f"blocks.{self.config.layer}.hook_resid_post"
            )
        if self.config.layer < 0:
            layer = self.model.cfg.n_layers + self.config.layer
        activations = cache[f"blocks.{layer}.hook_resid_post"]
        return activations.to(dtype=self.config.dtype)
    
    def get_activations_cache_path(self) -> str:
        """Get the path for caching model activations."""
        cache_name = (f"activations_"
                     f"layer{self.config.layer}_"
                     f"ctx{self.config.ctx_len}")
        return os.path.join(self.config.output_dir, "cache", cache_name)

    def load_activations_chunk(self, chunk_idx: int) -> Optional[torch.Tensor]:
        """Load a specific chunk of activations."""
        base_path = self.get_activations_cache_path()
        chunk_path = f"{base_path}_chunk{chunk_idx}.pt"
        
        if not os.path.exists(chunk_path):
            return None
            
        try:
            # Try to load the chunk
            chunk = torch.load(chunk_path)
            
            # Verify it's a tensor with the expected shape
            if not isinstance(chunk, torch.Tensor):
                print(f"Warning: Chunk {chunk_idx} is not a tensor, regenerating...")
                return None
                
            # Verify shape
            expected_shape = (self.config.chunk_size, self.config.ctx_len, self.model.cfg.d_model)
            if chunk.shape != expected_shape:
                print(f"Warning: Chunk {chunk_idx} has wrong shape {chunk.shape}, expected {expected_shape}, regenerating...")
                return None
                
            return chunk
            
        except Exception as e:
            print(f"Warning: Error loading chunk {chunk_idx}: {str(e)}")
            return None

    def get_num_chunks(self) -> int:
        """Get the number of activation chunks."""
        chunk_idx = 0
        while True:
            if not os.path.exists(f"{self.get_activations_cache_path()}_chunk{chunk_idx}.pt"):
                break
            chunk_idx += 1
        return chunk_idx

    def compute_chunk_metrics(
        self,
        probe: NgramProbe,
        chunk_logits: List[torch.Tensor],
        chunk_labels: List[torch.Tensor],
        ngrams: List[Tuple[int, ...]],
        ngram_to_idx: Dict[Tuple[int, ...], int]
    ) -> Dict[Tuple[int, ...], float]:
        """Compute metrics for a chunk of predictions, returning AUROC scores for each n-gram."""
        # Concatenate all batches
        all_logits = torch.cat(chunk_logits, dim=0)  # [total_samples, seq_len, num_ngrams]
        all_labels = torch.cat(chunk_labels, dim=0)   # [total_samples, seq_len, num_ngrams]
        
        # Flatten batch and sequence dimensions
        batch_size, seq_len, num_ngrams = all_logits.shape
        flat_logits = all_logits.reshape(-1, num_ngrams)  # [(batch_size * seq_len), num_ngrams]
        flat_labels = all_labels.reshape(-1, num_ngrams)   # [(batch_size * seq_len), num_ngrams]
        
        # Convert logits to probabilities
        probs = torch.sigmoid(flat_logits)
        
        # Transpose to get shape [num_ngrams, num_samples] as required by BinaryAUROC
        probs = probs.T
        labels = flat_labels.T
        
        # Create metric
        metric = BinaryAUROC(num_tasks=num_ngrams, device=self.config.device)
        
        # Update and compute
        metric.update(probs, labels)
        auroc_scores = metric.compute()
        
        # Create dictionary mapping n-grams to their AUROC scores
        results = {ngram: score.item() for ngram, score in zip(ngrams, auroc_scores)}
        
        # Filter out invalid scores (NaN or Inf)
        valid_results = {ngram: score for ngram, score in results.items() 
                        if not (np.isnan(score) or np.isinf(score))}
        
        if len(valid_results) < len(results):
            print(f"Warning: {len(results) - len(valid_results)} n-grams had invalid AUROC scores")
        
        return valid_results

    def prepare_probe_data(
        self,
        tokens: torch.Tensor,
        activations: torch.Tensor,
        ngrams: List[Tuple[int, ...]],
        ngram_to_idx: Dict[Tuple[int, ...], int],
        probe_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for a specific probe type (first or final position)."""
        batch_size, seq_len, d_model = activations.shape
        
        # Initialize labels tensor with zeros
        labels = torch.zeros((batch_size, seq_len, len(ngrams)), 
                           device=self.config.device,
                           dtype=self.config.dtype)
        
        # For each sequence in the batch
        for b in range(batch_size):
            sequence = tokens[b].cpu().tolist()
            
            # For each possible n-gram position in the sequence
            for j in range(seq_len - self.config.ngram_size + 1):
                # Get the current n-gram
                current_ngram = tuple(sequence[j:j + self.config.ngram_size])
                
                # Check if this n-gram is part of any set
                for set_idx, ngram_set in enumerate(ngrams):
                    if isinstance(ngram_set, tuple):
                        if len(ngram_set) > 0 and isinstance(ngram_set[0], tuple):
                            # This is a set of n-grams
                            for ngram in ngram_set:
                                if current_ngram == ngram:
                                    # For first position probe, use the start position
                                    # For final position probe, use the end position
                                    pos = j if probe_type == 'first' else j + self.config.ngram_size - 1
                                    labels[b, pos, set_idx] = 1
                                    break
                        else:
                            # This is a single n-gram
                            if current_ngram == ngram_set:
                                pos = j if probe_type == 'first' else j + self.config.ngram_size - 1
                                labels[b, pos, set_idx] = 1
        
        # Determine valid positions based on probe type
        if probe_type == 'first':
            # For first position, we can use all positions up to seq_len - ngram_size + 1
            valid_positions = list(range(seq_len - self.config.ngram_size + 1))
        else:
            # For final position, we can only use positions from ngram_size - 1 to seq_len
            valid_positions = list(range(self.config.ngram_size - 1, seq_len))
        
        # Select only valid positions from activations and labels
        activations = activations[:, valid_positions, :].to(self.config.device)
        labels = labels[:, valid_positions, :]
        
        return activations, labels

    def train_probe(
        self,
        tokens: Optional[torch.Tensor],
        ngrams: List[Tuple[int, ...]],
        ngram_to_idx: Dict[Tuple[int, ...], int]
    ) -> Dict[str, Dict[str, float]]:
        """Train final-position probe using chunked activations."""
        # Count total tokens across all chunks
        total_tokens = 0
        chunk_idx = 0
        while True:
            chunk = self.load_tokens_chunk(chunk_idx)
            if chunk is None:
                break
            total_tokens += len(chunk)
            chunk_idx += 1
        
        if chunk_idx == 0:
            raise ValueError("No token chunks found! Run get_tokens() first.")
        
        total_tokens_with_context = total_tokens * self.config.ctx_len
        required_tokens = self.config.num_train_tokens + self.config.num_val_tokens
        if required_tokens > total_tokens_with_context:
            raise ValueError(f"Not enough tokens available ({total_tokens_with_context} < {required_tokens})")
            
        # Calculate total number of possible n-gram positions
        total_positions = total_tokens_with_context - (self.config.ngram_size - 1) * total_tokens
        
        print("\nStarting probe training:")
        print(f"Number of n-grams selected: {len(ngrams)}")
        print(f"First few n-grams: {ngrams[:5]}")
        print(f"N-gram types: {[type(ng) for ng in ngrams[:5]]}")
        print(f"N-gram lengths: {[len(ng) for ng in ngrams[:5]]}")
        print(f"Frequency range: {min(self.ngram_counts.values()) / total_positions:.2e} to {max(self.ngram_counts.values()) / total_positions:.2e}")
        
        # Create frequency vs rank plot
        print("\nCreating frequency vs rank plot...")
        plt.figure(figsize=(12, 6))
        
        # Get all counts and sort them in descending order
        counts = sorted(self.ngram_counts.values(), reverse=True)
        frequencies = [count / total_positions for count in counts]
        ranks = list(range(1, len(frequencies) + 1))
        
        plt.plot(ranks, frequencies, 'b-', alpha=0.7, linewidth=1, label='All n-grams')
        
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.title(f"N-gram Frequency Distribution (n={self.config.ngram_size})")
        plt.xlabel("Rank (log scale)")
        plt.ylabel("Frequency (log scale)")
        
        # Add horizontal lines for frequency thresholds
        plt.axhline(y=1e-2, color='r', linestyle='--', alpha=0.5, label='1e-2 threshold')
        plt.axhline(y=1e-4, color='g', linestyle='--', alpha=0.5, label='1e-4 threshold')
        
        plt.legend()
        
        # Save plot
        os.makedirs(os.path.join(self.config.output_dir, "plots"), exist_ok=True)
        plot_path = os.path.join(self.config.output_dir, "plots", "ngram_frequency_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved frequency distribution plot to {plot_path}")
        
        # Print summary statistics
        print(f"\nTraining on {len(ngrams)} n-grams")
        print(f"Most frequent n-gram appears in {frequencies[0]:.2e} of positions")
        print(f"Least frequent n-gram appears in {frequencies[-1]:.2e} of positions")
        
        # Calculate number of sequences needed for training and validation
        num_train_sequences = math.ceil(self.config.num_train_tokens / self.config.ctx_len)
        num_val_sequences = math.ceil(self.config.num_val_tokens / self.config.ctx_len)
        
        # Calculate number of chunks for each split
        num_train_chunks = math.ceil(num_train_sequences / self.config.chunk_size)
        num_val_chunks = math.ceil(num_val_sequences / self.config.chunk_size)
        
        print(f"\nSplit configuration:")
        print(f"Training: {num_train_sequences} sequences in {num_train_chunks} chunks")
        print(f"Validation: {num_val_sequences} sequences in {num_val_chunks} chunks")
        
        # Create probe for final position
        probe = NgramProbe(
            d_model=self.model.cfg.d_model,
            ngrams=ngrams,
            device=self.config.device,
            probe_type='final',
            dtype=self.config.dtype
        ).to(self.config.device)
        
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        print("\nTraining probe...")
        train_activations = []
        train_labels = []
        
        # Calculate total number of batches for progress bar
        total_batches = 0
        for chunk_idx in range(num_train_chunks):
            chunk_tokens = self.load_tokens_chunk(chunk_idx)
            if chunk_tokens is None:
                continue
            total_batches += (len(chunk_tokens) + self.config.batch_size - 1) // self.config.batch_size
        
        # Create progress bar
        pbar = tqdm(total=total_batches, desc="Training batches")
        current_batch = 0
        
        for chunk_idx in range(num_train_chunks):
            # Load both tokens and activations for this chunk
            chunk_tokens = self.load_tokens_chunk(chunk_idx)
            if chunk_tokens is None:
                print(f"Warning: Could not load token chunk {chunk_idx}, skipping...")
                continue
                
            activations = self.load_activations_chunk(chunk_idx)
            if activations is None:
                print(f"Warning: Could not load activation chunk {chunk_idx}, skipping...")
                continue
                
            # Verify alignment between tokens and activations
            if not self.verify_chunk_alignment(chunk_tokens, activations, chunk_idx):
                print(f"Warning: Chunk {chunk_idx} has alignment issues, skipping...")
                continue
            
            for i in range(0, len(chunk_tokens), self.config.batch_size):
                batch_end = min(i + self.config.batch_size, len(chunk_tokens))
                batch_tokens = chunk_tokens[i:batch_end]
                batch_activations = activations[i:batch_end]
                
                activations_batch, labels = self.prepare_probe_data(
                    batch_tokens, batch_activations, ngrams, ngram_to_idx, 'final'
                )
                
                logits = probe(activations_batch)
                loss = probe.compute_loss(logits, labels, self.config.positive_weight)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_activations.append(activations_batch.cpu())
                train_labels.append(labels.cpu())
                
                # Update progress bar with loss and label statistics
                current_batch += 1
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pos_labels': f'{(labels > 0).sum().item()}'
                })
                pbar.update(1)
                
                del activations_batch, labels, logits
                torch.cuda.empty_cache()
            
            del activations, chunk_tokens
            torch.cuda.empty_cache()
        
        pbar.close()
        
        # Validation
        probe.eval()
        
        print("\nEvaluating final-position probe...")
        print(f"Number of validation chunks to evaluate: {num_val_chunks}")
        auroc_accumulator = {ngram: [] for ngram in ngrams}
        
        for chunk_idx in range(num_val_chunks):
            print(f"\nProcessing validation chunk {chunk_idx + 1}/{num_val_chunks}")
            
            # Load both tokens and activations for this chunk
            chunk_tokens = self.load_tokens_chunk(chunk_idx + num_train_chunks)
            if chunk_tokens is None:
                print(f"Warning: Could not load validation token chunk {chunk_idx + num_train_chunks}, skipping...")
                continue
                
            activations = self.load_activations_chunk(chunk_idx + num_train_chunks)
            if activations is None:
                print(f"Warning: Could not load validation activation chunk {chunk_idx + num_train_chunks}, skipping...")
                continue
                
            # Verify alignment between tokens and activations
            if not self.verify_chunk_alignment(chunk_tokens, activations, chunk_idx + num_train_chunks):
                print(f"Warning: Validation chunk {chunk_idx + num_train_chunks} has alignment issues, skipping...")
                continue
                
            chunk_logits = []
            chunk_labels = []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(chunk_tokens), self.config.batch_size)):
                    batch_end = min(i + self.config.batch_size, len(chunk_tokens))
                    batch_tokens = chunk_tokens[i:batch_end]
                    batch_activations = activations[i:batch_end]
                    
                    # Prepare data for the batch
                    activations_batch, labels = self.prepare_probe_data(
                        batch_tokens, batch_activations, ngrams, ngram_to_idx, 'final'
                    )
                    
                    # Get predictions
                    logits = probe(activations_batch)
                    
                    # Store predictions and labels
                    chunk_logits.append(logits.cpu())
                    chunk_labels.append(labels.cpu())
                    
                    del activations_batch, labels, logits
                    torch.cuda.empty_cache()
            
            # Compute metrics for this chunk
            print(f"\nComputing metrics for chunk {chunk_idx}...")
            chunk_results = self.compute_chunk_metrics(probe, chunk_logits, chunk_labels, ngrams, ngram_to_idx)
            
            # Accumulate AUROC scores
            for ngram in ngrams:
                if ngram in chunk_results:
                    auroc_accumulator[ngram].append(chunk_results[ngram])
            
            print(f"Completed chunk {chunk_idx} evaluation")
            del activations, chunk_tokens, chunk_logits, chunk_labels, chunk_results
            torch.cuda.empty_cache()
        
        print("\nComputing final probe results...")
        # Only compute mean for n-grams that have valid scores from at least one chunk
        results = {
            ngram: np.mean(scores) 
            for ngram, scores in auroc_accumulator.items() 
            if len(scores) > 0
        }
        
        # Print summary statistics
        auroc_scores = list(results.values())
        if auroc_scores:
            print(f"Number of n-grams with valid AUROC scores: {len(auroc_scores)}")
            print(f"Mean AUROC score: {np.mean(auroc_scores):.4f}")
            print(f"Min AUROC score: {np.min(auroc_scores):.4f}")
            print(f"Max AUROC score: {np.max(auroc_scores):.4f}")
        else:
            print("Warning: No valid AUROC scores were computed!")
        
        # Save final model
        final_model_path = os.path.join(self.config.output_dir, "probe_final.pt")
        probe.save(final_model_path)
        print(f"Saved final model to {final_model_path}")
        
        return {'final': results}
    
    def save_results(self, results: Dict[str, Dict[str, float]]):
        """Save results and create plots."""
        # Convert tuple keys to strings for JSON serialization
        json_results = {
            'final': {
                str([list(ngram) if isinstance(ngram, tuple) else list([ngram]) for ngram in ngram_set]): score
                for ngram_set, score in results['final'].items()
            }
        }
        
        # Save raw results
        output_file = os.path.join(self.config.output_dir, "probe_results.json")
        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2)
        
        # Create plots directory path
        plots_dir = os.path.join(self.config.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract data for plotting
        log_errors = np.log10(1 - np.array(list(results['final'].values())))
        
        # Calculate total tokens from chunks
        total_tokens = 0
        chunk_idx = 0
        while True:
            chunk = self.load_tokens_chunk(chunk_idx)
            if chunk is None:
                break
            total_tokens += len(chunk)
            chunk_idx += 1
        
        if chunk_idx == 0:
            print("Warning: No token chunks found for plotting!")
            return
        
        total_tokens_with_context = total_tokens * self.config.ctx_len
        counts = np.array([self.ngram_counts[ngram_set] for ngram_set in results['final'].keys()])
        frequencies = counts / total_tokens_with_context
        log_frequencies = np.log10(frequencies)
        
        # Create histogram
        plt.figure(figsize=(12, 6))
        
        # Calculate bins based on log error rates
        valid_log_errors = log_errors[~np.isnan(log_errors) & ~np.isinf(log_errors)]
        
        if len(valid_log_errors) > 0:
            bins = np.linspace(min(valid_log_errors), max(valid_log_errors), self.config.histogram_bins)
            
            plt.hist(valid_log_errors, bins=bins, alpha=0.5, label='Final Position')
            plt.title("Distribution of Log10 Error Rates (log10(1 - AUROC))")
            plt.xlabel("Log10 Error Rate")
            plt.ylabel("Count")
            plt.legend()
            plt.savefig(os.path.join(plots_dir, f"log_error_rate_histogram_n{self.config.ngram_size}_union{self.config.union_size}.png"))
        plt.close()
        
        # Create scatter plot with normalized frequencies
        plt.figure(figsize=(12, 6))
        
        # Filter out invalid values
        valid_mask = ~np.isnan(log_frequencies) & ~np.isnan(log_errors) & \
                    ~np.isinf(log_frequencies) & ~np.isinf(log_errors)
        
        if np.sum(valid_mask) > 0:
            plt.scatter(log_frequencies[valid_mask], log_errors[valid_mask], alpha=0.5, label='Final Position')
            plt.title("Log10 Error Rate vs Log10 Frequency")
            plt.xlabel("Log10 Frequency (log10(occurrences / total tokens))")
            plt.ylabel("Log10 Error Rate (log10(1 - AUROC))")
            plt.legend()
            
            # Add trend line with R² value
            valid_x = log_frequencies[valid_mask]
            valid_y = log_errors[valid_mask]
            
            if len(valid_x) > 1 and len(valid_y) > 1:
                z = np.polyfit(valid_x, valid_y, 1)
                p = np.poly1d(z)
                r2 = np.corrcoef(valid_x, valid_y)[0,1]**2
                
                x_range = np.linspace(valid_x.min(), valid_x.max(), 100)
                plt.plot(x_range, p(x_range), 'r--', alpha=0.8, label=f'Trend (R² = {r2:.3f})')
            
            plt.legend()
            plt.savefig(os.path.join(plots_dir, f"log_error_rate_vs_freq_n{self.config.ngram_size}_union{self.config.union_size}.png"))
        plt.close()

    def generate_and_save_activations(self) -> None:
        """Generate activations for all tokens and save them to disk in chunks."""
        cache_dir = os.path.join(self.config.output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_path = self.get_activations_cache_path()
        
        # Count number of token chunks
        chunk_idx = 0
        while os.path.exists(f"{os.path.join(cache_dir, 'cached_tokens')}_chunk{chunk_idx}.pt"):
            chunk_idx += 1
        
        if chunk_idx == 0:
            raise ValueError("No token chunks found! Run get_tokens() first.")
        
        print(f"\nGenerating activations for {chunk_idx} token chunks...")
        
        for chunk_idx in range(chunk_idx):
            # Load token chunk
            chunk_tokens = self.load_tokens_chunk(chunk_idx)
            if chunk_tokens is None:
                print(f"Warning: Could not load token chunk {chunk_idx}, skipping...")
                continue
            
            chunk_path = f"{base_path}_chunk{chunk_idx}.pt"
            
            # Skip if chunk exists and is valid
            if os.path.exists(chunk_path):
                try:
                    chunk = torch.load(chunk_path)
                    if isinstance(chunk, torch.Tensor) and chunk.shape == (len(chunk_tokens), self.config.ctx_len, self.model.cfg.d_model):
                        print(f"Chunk {chunk_idx} already exists and is valid, skipping...")
                        continue
                except Exception:
                    print(f"Chunk {chunk_idx} exists but is corrupted, regenerating...")
            
            print(f"\nProcessing chunk {chunk_idx + 1}/{chunk_idx} (sequences 0 to {len(chunk_tokens)})")
            chunk_activations = []
            
            # Process the chunk in batches
            for i in tqdm(range(0, len(chunk_tokens), self.config.model_batch_size)):
                batch_end = min(i + self.config.model_batch_size, len(chunk_tokens))
                batch_tokens = chunk_tokens[i:batch_end]
                
                with torch.no_grad():
                    activations = self.get_activations_batch(batch_tokens)
                    chunk_activations.append(activations.cpu())
                
                del activations
                torch.cuda.empty_cache()
            
            # Save this chunk
            chunk_activations = torch.cat(chunk_activations, dim=0)
            torch.save(chunk_activations, chunk_path)
            print(f"Saved chunk {chunk_idx} with shape {chunk_activations.shape}")
            
            del chunk_activations, chunk_tokens
            torch.cuda.empty_cache()

    def load_tokens_chunk(self, chunk_idx: int) -> Optional[torch.Tensor]:
        """Load a specific chunk of tokens."""
        cache_dir = os.path.join(self.config.output_dir, "cache")
        base_path = os.path.join(cache_dir, "cached_tokens")
        chunk_path = f"{base_path}_chunk{chunk_idx}.pt"
        
        if not os.path.exists(chunk_path):
            return None
        
        try:
            # Try to load the chunk
            chunk = torch.load(chunk_path)
            
            # Verify it's a tensor with the expected shape
            if not isinstance(chunk, torch.Tensor):
                print(f"Warning: Chunk {chunk_idx} is not a tensor, regenerating...")
                return None
            
            # Verify shape - allow for variable chunk sizes
            if len(chunk.shape) != 2:
                print(f"Warning: Chunk {chunk_idx} has wrong number of dimensions: {len(chunk.shape)}, expected 2")
                return None
            if chunk.shape[1] != self.config.ctx_len:
                print(f"Warning: Chunk {chunk_idx} has wrong sequence length: {chunk.shape[1]}, expected {self.config.ctx_len}")
                return None
            
            return chunk
            
        except Exception as e:
            print(f"Warning: Error loading chunk {chunk_idx}: {str(e)}")
            return None

    def verify_chunk_alignment(self, token_chunk: torch.Tensor, activation_chunk: torch.Tensor, chunk_idx: int) -> bool:
        """Verify that a token chunk and activation chunk are properly aligned."""
        # Check number of dimensions
        if len(token_chunk.shape) != 2 or len(activation_chunk.shape) != 3:
            print(f"Warning: Chunk {chunk_idx} has wrong number of dimensions: tokens={len(token_chunk.shape)}, activations={len(activation_chunk.shape)}")
            return False
        
        # Check batch size (number of sequences)
        if token_chunk.shape[0] != activation_chunk.shape[0]:
            print(f"Warning: Chunk {chunk_idx} has mismatched batch sizes: tokens={token_chunk.shape[0]}, activations={activation_chunk.shape[0]}")
            return False
        
        # Check sequence length
        if token_chunk.shape[1] != activation_chunk.shape[1]:
            print(f"Warning: Chunk {chunk_idx} has mismatched sequence lengths: tokens={token_chunk.shape[1]}, activations={activation_chunk.shape[1]}")
            return False
        
        # Check activation dimension
        if activation_chunk.shape[2] != self.model.cfg.d_model:
            print(f"Warning: Chunk {chunk_idx} has wrong activation dimension: {activation_chunk.shape[2]}, expected {self.model.cfg.d_model}")
            return False
        
        return True

def main():
    config = NgramProbingConfig()
    
    try:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "cache"), exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        return
    
    # Create a prober for initial setup and activation generation
    prober = NgramProber(config)
    dataset = prober.load_dataset()
    
    # Get tokens (now returns None as we're using chunked loading)
    prober.get_tokens(dataset)
    
    # Check if activations already exist in the common cache directory
    base_path = prober.get_activations_cache_path()
    chunk_idx = 0
    while os.path.exists(f"{base_path}_chunk{chunk_idx}.pt"):
        chunk_idx += 1
    
    if chunk_idx == 0:
        # No existing activations found, generate them
        print("\nNo existing activations found. Generating activations (will be reused for all n-gram sizes)...")
        prober.generate_and_save_activations()
    else:
        print(f"\nFound existing activations ({chunk_idx} chunks). Skipping generation.")
    
    # Run for different n-gram sizes
    for n in range(4, 10):
        print(f"\n{'='*50}")
        print(f"Running probing for n-gram size {n}")
        print(f"{'='*50}")
        
        # Create a new config for this n-gram size
        current_config = NgramProbingConfig()
        current_config.ngram_size = n
        current_config.output_dir = os.path.join(config.output_dir, f"n{n}")
        
        # Create necessary directories for this n-gram size
        os.makedirs(current_config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(current_config.output_dir, "plots"), exist_ok=True)
        
        # Create a new prober for this n-gram size
        current_prober = NgramProber(current_config)
        
        # Override the cache directory to use the common one
        current_prober.config.output_dir = config.output_dir
        
        # Get n-grams and train probe
        # Note: We pass None as tokens since we're using chunked loading
        ngrams = current_prober.get_ngrams(None)
        ngram_list = list(ngrams.keys())
        ngram_to_idx = {ngram: idx for idx, ngram in enumerate(ngram_list)}
        
        # Train probe using the pre-generated activations
        results = current_prober.train_probe(None, ngram_list, ngram_to_idx)
        current_prober.save_results(results)
        
        print(f"\nCompleted n-gram size {n}")
        print(f"Results saved to {current_config.output_dir}")
        
        # Clear memory
        del current_prober, ngrams, ngram_list, ngram_to_idx, results
        torch.cuda.empty_cache()
    
    print("\nCompleted all n-gram sizes!")

if __name__ == "__main__":
    main() 