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
        
        # Check if all chunks exist
        if not force_recompute:
            chunk_idx = 0
            all_chunks_exist = True
            while True:
                chunk_path = f"{base_path}_chunk{chunk_idx}.pt"
                if not os.path.exists(chunk_path):
                    break
                chunk_idx += 1
            if chunk_idx > 0:
                # Load all chunks
                chunks = []
                for i in range(chunk_idx):
                    chunk_path = f"{base_path}_chunk{i}.pt"
                    chunks.append(torch.load(chunk_path))
                self.tokens = torch.cat(chunks, dim=0)
                return self.tokens
        
        # Tokenize in chunks
        chunk_size = 10000  # Process 10k texts at a time
        num_chunks = (len(dataset) + chunk_size - 1) // chunk_size
        all_tokens = []
        
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
            
            all_tokens.append(chunk_tokens)
            
            # Clear memory
            del chunk_tokens
            torch.cuda.empty_cache()
        
        # Combine all chunks
        self.tokens = torch.cat(all_tokens, dim=0)
        return self.tokens
        
    def get_ngrams(self, tokens: torch.Tensor) -> Dict[str, int]:
        """Extract n-grams from tokenized texts."""
        if self.load_ngram_counts():
            return self.ngram_counts
            
        # Calculate total number of possible n-gram positions
        total_tokens = len(tokens) * self.config.ctx_len
        total_positions = total_tokens - (self.config.ngram_size - 1) * len(tokens)
        
        # Use running counter with frequency-based filtering
        ngram_counts = Counter()
        min_count = int(1e-5 * total_positions)  # Minimum count needed to meet frequency threshold
        max_count = int(1e-2 * total_positions)  # Maximum count allowed by frequency threshold
        
        print(f"\nCounting n-grams (filtering counts between {min_count} and {max_count})...")
        for batch_start in tqdm(range(0, len(tokens), self.config.model_batch_size), desc="Finding n-grams"):
            batch = tokens[batch_start:batch_start + self.config.model_batch_size].cpu().numpy()
            
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
        
        # Filter n-grams by frequency thresholds
        selected_ngrams = []
        for ngram, count in ngram_counts.most_common():
            if min_count <= count <= max_count:
                selected_ngrams.append((ngram, count))
                if len(selected_ngrams) >= self.config.top_m_ngrams:
                    break
        
        # Store selected n-grams
        self.ngram_counts = dict(selected_ngrams)
        
        # Print information about selected n-grams
        print(f"\nSelected {len(selected_ngrams)} n-grams:")
        if selected_ngrams:
            min_freq = selected_ngrams[-1][1] / total_positions
            max_freq = selected_ngrams[0][1] / total_positions
            print(f"Frequency range: {min_freq:.2e} to {max_freq:.2e}")
        
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
        return {ngram: score.item() for ngram, score in zip(ngrams, auroc_scores)}

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
        
        labels = torch.zeros((batch_size, seq_len, len(ngrams)), 
                           device=self.config.device,
                           dtype=self.config.dtype)
        
        for b in range(batch_size):
            for j in range(seq_len - self.config.ngram_size + 1):
                current_ngram = tuple(tokens[b, j:j + self.config.ngram_size].cpu().tolist())
                if current_ngram in ngram_to_idx:
                    pos = j if probe_type == 'first' else j + self.config.ngram_size - 1
                    labels[b, pos, ngram_to_idx[current_ngram]] = 1
        
        if probe_type == 'first':
            valid_positions = list(range(seq_len - self.config.ngram_size + 1))
        else:
            valid_positions = list(range(self.config.ngram_size - 1, seq_len))
            
        activations = activations[:, valid_positions, :].to(self.config.device)
        labels = labels[:, valid_positions, :]
        
        return activations, labels

    def train_probe(
        self,
        tokens: torch.Tensor,
        ngrams: List[Tuple[int, ...]],
        ngram_to_idx: Dict[Tuple[int, ...], int]
    ) -> Dict[str, Dict[str, float]]:
        """Train final-position probe using chunked activations."""
        total_tokens = len(tokens) * self.config.ctx_len
        required_tokens = self.config.num_train_tokens + self.config.num_val_tokens
        if required_tokens > total_tokens:
            raise ValueError(f"Not enough tokens available ({total_tokens} < {required_tokens})")
            
        # Calculate total number of possible n-gram positions
        total_positions = total_tokens - (self.config.ngram_size - 1) * len(tokens)
        
        print("\nStarting probe training:")
        print(f"Number of n-grams selected: {len(ngrams)}")
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
        
        # Calculate number of chunks needed for training and validation
        num_train_sequences = math.ceil(self.config.num_train_tokens / self.config.ctx_len)
        num_val_sequences = math.ceil(self.config.num_val_tokens / self.config.ctx_len)
        num_train_chunks = math.ceil(num_train_sequences / self.config.chunk_size)
        num_val_chunks = math.ceil(num_val_sequences / self.config.chunk_size)
        
        # Create probe for final position
        probe = NgramProbe(
            d_model=self.model.cfg.d_model,
            ngrams=ngrams,
            device=self.config.device,
            probe_type='final',
            dtype=self.config.dtype
        ).to(self.config.device)
        
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.config.learning_rate)
        
        # Load and prepare training data
        print("\nLoading training data for final-position probe...")
        train_activations = []
        train_labels = []
        
        # Calculate total number of batches for progress bar
        total_batches = 0
        for chunk_idx in range(num_train_chunks):
            chunk_tokens = tokens[chunk_idx * self.config.chunk_size:(chunk_idx + 1) * self.config.chunk_size]
            total_batches += (len(chunk_tokens) + self.config.batch_size - 1) // self.config.batch_size
        
        # Create progress bar
        pbar = tqdm(total=total_batches, desc="Training batches")
        current_batch = 0
        
        for chunk_idx in range(num_train_chunks):
            activations = self.load_activations_chunk(chunk_idx)
            if activations is None:
                continue
            
            chunk_tokens = tokens[chunk_idx * self.config.chunk_size:(chunk_idx + 1) * self.config.chunk_size]
            
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
                
                # Update progress bar with loss
                current_batch += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
                
                del activations_batch, labels, logits
                torch.cuda.empty_cache()
            
            del activations
            torch.cuda.empty_cache()
        
        pbar.close()
        
        # Validation
        probe.eval()
        
        print("\nEvaluating final-position probe...")
        print(f"Number of chunks to evaluate: {self.get_num_chunks()}")
        auroc_accumulator = {ngram: [] for ngram in ngrams}
        
        for chunk_idx in range(self.get_num_chunks()):
            print(f"\nProcessing evaluation chunk {chunk_idx + 1}/{self.get_num_chunks()}")
            activations = self.load_activations_chunk(chunk_idx)
            if activations is None:
                print(f"Warning: Could not load chunk {chunk_idx}, skipping...")
                continue
                
            chunk_tokens = tokens[chunk_idx * self.config.chunk_size:(chunk_idx + 1) * self.config.chunk_size]
            print(f"Chunk tokens shape: {chunk_tokens.shape}")
            chunk_logits = []
            chunk_labels = []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(chunk_tokens), self.config.batch_size)):
                    batch_end = min(i + self.config.batch_size, len(chunk_tokens))
                    batch_tokens = chunk_tokens[i:batch_end]
                    batch_activations = activations[i:batch_end]
                    
                    activations_batch, labels = self.prepare_probe_data(
                        batch_tokens, batch_activations, ngrams, ngram_to_idx, 'final'
                    )
                    logits = probe(activations_batch)
                    
                    chunk_logits.append(logits.cpu())
                    chunk_labels.append(labels.cpu())
                    
                    del activations_batch, labels, logits
                    torch.cuda.empty_cache()
            
            print(f"Number of logit batches: {len(chunk_logits)}")
            print(f"Number of label batches: {len(chunk_labels)}")
            
            # Compute metrics for this chunk
            print(f"\nComputing metrics for chunk {chunk_idx}...")
            chunk_results = self.compute_chunk_metrics(probe, chunk_logits, chunk_labels, ngrams, ngram_to_idx)
            
            # Accumulate AUROC scores
            for ngram in ngrams:
                auroc_accumulator[ngram].append(chunk_results[ngram])
            
            print(f"Completed chunk {chunk_idx} evaluation")
            del activations, chunk_logits, chunk_labels, chunk_results
            torch.cuda.empty_cache()
        
        print("\nComputing final probe results...")
        results = {
            ngram: np.mean(auroc_accumulator[ngram]) 
            for ngram in ngrams
        }
        
        # Print summary statistics
        auroc_scores = list(results.values())
        print(f"Number of n-grams with valid AUROC scores: {len(auroc_scores)}")
        print(f"Mean AUROC score: {np.mean(auroc_scores):.4f}")
        print(f"Min AUROC score: {np.min(auroc_scores):.4f}")
        print(f"Max AUROC score: {np.max(auroc_scores):.4f}")
        
        # Save final model
        final_model_path = os.path.join(self.config.output_dir, "probe_final.pt")
        probe.save(final_model_path)
        print(f"Saved final model to {final_model_path}")
        
        return {'final': results}
    
    def save_results(self, results: Dict[str, Dict[Tuple[int, ...], float]]):
        """Save results and create plots."""
        # Convert tuple keys to strings for JSON serialization
        json_results = {
            'final': {
                str(list(ngram)): score  # Convert tuple to list, then to string
                for ngram, score in results['final'].items()
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
        
        # Calculate normalized frequencies
        total_tokens = len(self.tokens) * self.config.ctx_len
        counts = np.array([self.ngram_counts[ngram] for ngram in results['final'].keys()])
        frequencies = counts / total_tokens
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
            plt.savefig(os.path.join(plots_dir, f"log_error_rate_histogram_n{self.config.ngram_size}.png"))
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
            plt.savefig(os.path.join(plots_dir, f"log_error_rate_vs_freq_n{self.config.ngram_size}.png"))
        plt.close()

    def generate_and_save_activations(self, tokens: torch.Tensor) -> None:
        """Generate activations for all tokens and save them to disk in chunks."""
        cache_dir = os.path.join(self.config.output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_path = self.get_activations_cache_path()
        
        print("\nGenerating activations in chunks...")
        chunk_size = min(self.config.chunk_size, len(tokens))  # Process chunk_size sequences at a time
        num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(tokens))
            
            chunk_path = f"{base_path}_chunk{chunk_idx}.pt"
            
            # Skip if chunk exists and is valid
            if os.path.exists(chunk_path):
                try:
                    chunk = torch.load(chunk_path)
                    if isinstance(chunk, torch.Tensor) and chunk.shape == (end_idx - start_idx, self.config.ctx_len, self.model.cfg.d_model):
                        print(f"Chunk {chunk_idx} already exists and is valid, skipping...")
                        continue
                except Exception:
                    print(f"Chunk {chunk_idx} exists but is corrupted, regenerating...")
            
            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (sequences {start_idx} to {end_idx})")
            chunk_activations = []
            
            # Process the chunk in batches
            for i in tqdm(range(start_idx, end_idx, self.config.model_batch_size)):
                batch_end = min(i + self.config.model_batch_size, end_idx)
                batch_tokens = tokens[i:batch_end]
                
                with torch.no_grad():
                    activations = self.get_activations_batch(batch_tokens)
                    chunk_activations.append(activations.cpu())
                
                del activations
                torch.cuda.empty_cache()
            
            # Save this chunk
            chunk_activations = torch.cat(chunk_activations, dim=0)
            torch.save(chunk_activations, chunk_path)
            print(f"Saved chunk {chunk_idx} with shape {chunk_activations.shape}")
            
            del chunk_activations
            torch.cuda.empty_cache()

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
    tokens = prober.get_tokens(dataset)
    
    # Check if activations already exist in the common cache directory
    base_path = prober.get_activations_cache_path()
    chunk_idx = 0
    while os.path.exists(f"{base_path}_chunk{chunk_idx}.pt"):
        chunk_idx += 1
    
    if chunk_idx == 0:
        # No existing activations found, generate them
        print("\nNo existing activations found. Generating activations (will be reused for all n-gram sizes)...")
        prober.generate_and_save_activations(tokens)
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
        current_prober.tokens = tokens  # Reuse the same tokens
        
        # Override the cache directory to use the common one
        current_prober.config.output_dir = config.output_dir
        
        # Get n-grams and train probe
        ngrams = current_prober.get_ngrams(tokens)
        ngram_list = list(ngrams.keys())
        ngram_to_idx = {ngram: idx for idx, ngram in enumerate(ngram_list)}
        
        # Train probe using the pre-generated activations
        results = current_prober.train_probe(tokens, ngram_list, ngram_to_idx)
        current_prober.save_results(results)
        
        print(f"\nCompleted n-gram size {n}")
        print(f"Results saved to {current_config.output_dir}")
        
        # Clear memory
        del current_prober, ngrams, ngram_list, ngram_to_idx, results
        torch.cuda.empty_cache()
    
    print("\nCompleted all n-gram sizes!")

if __name__ == "__main__":
    main() 