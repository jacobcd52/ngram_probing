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
                     f"top{self.config.top_m_ngrams}_"
                     f"ignore{self.config.ignore_top_n}.pkl")
        return os.path.join(self.config.output_dir, "cache", cache_name)
        
    def save_ngram_counts(self):
        """Save n-gram counts to a cache file."""
        if self.ngram_counts is None:
            raise ValueError("No n-gram counts to save!")
            
        cache_path = self.get_cache_path()
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'ngram_counts': self.ngram_counts,
                'ignored_ngrams': getattr(self, 'ignored_ngrams', {}),
                'config': {
                    'ngram_size': self.config.ngram_size,
                    'ctx_len': self.config.ctx_len,
                    'top_m_ngrams': self.config.top_m_ngrams,
                    'ignore_top_n': self.config.ignore_top_n
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
                cached_config['top_m_ngrams'] != self.config.top_m_ngrams or
                cached_config.get('ignore_top_n', 0) != self.config.ignore_top_n):
                return False
                
            self.ngram_counts = data['ngram_counts']
            if 'ignored_ngrams' in data:
                self.ignored_ngrams = data['ignored_ngrams']
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
        cache_path = os.path.join(self.config.output_dir, "cached_tokens.pt")
        
        if not force_recompute and os.path.exists(cache_path):
            self.tokens = torch.load(cache_path)
            return self.tokens
            
        self.tokens = utils.tokenize_and_concatenate(
            dataset, 
            self.model.tokenizer, 
            max_length=self.config.ctx_len,
            add_bos_token=False
        )['tokens']
        
        torch.save(self.tokens, cache_path)
        return self.tokens
        
    def get_ngrams(self, tokens: torch.Tensor) -> Dict[str, int]:
        """Extract n-grams from tokenized texts."""
        if self.load_ngram_counts():
            return self.ngram_counts
            
        # Use running counter instead of accumulating all n-grams
        ngram_counts = Counter()
        for batch_start in tqdm(range(0, len(tokens), self.config.model_batch_size), desc="Finding n-grams"):
            batch = tokens[batch_start:batch_start + self.config.model_batch_size].cpu().numpy()
            
            for seq in batch:
                seq_ngrams = [
                    tuple(seq[j:j + self.config.ngram_size].tolist())
                    for j in range(len(seq) - self.config.ngram_size + 1)
                ]
                # Update counter directly instead of accumulating
                ngram_counts.update(seq_ngrams)
            
            del batch
            if batch_start % (5 * self.config.model_batch_size) == 0:
                torch.cuda.empty_cache()
        
        total_positions = sum(ngram_counts.values())
        
        # Create frequency vs rank plot
        print("\nCreating frequency vs rank plot...")
        plt.figure(figsize=(12, 6))
        
        # Get all counts and sort them in descending order
        counts = sorted(ngram_counts.values(), reverse=True)
        frequencies = [count / total_positions * 100 for count in counts]
        ranks = list(range(1, len(frequencies) + 1))
        
        plt.plot(ranks, frequencies, 'b-', alpha=0.7, linewidth=1)
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.title(f"N-gram Frequency Distribution (n={self.config.ngram_size})")
        plt.xlabel("Rank (log scale)")
        plt.ylabel("Frequency % (log scale)")
        
        # Add vertical lines for frequency thresholds
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
        plt.axhline(y=0.01, color='g', linestyle='--', alpha=0.5, label='0.01% threshold')
        
        plt.legend()
        
        # Save plot
        os.makedirs(os.path.join(self.config.output_dir, "plots"), exist_ok=True)
        plot_path = os.path.join(self.config.output_dir, "plots", "ngram_frequency_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved frequency distribution plot to {plot_path}")
        
        # Select n-grams based on frequency thresholds
        selected_ngrams = []
        for ngram, count in ngram_counts.most_common():
            freq = count / total_positions * 100
            if 0.01 <= freq <= 1.0:
                selected_ngrams.append((ngram, count))
                if len(selected_ngrams) >= 1000:
                    break
        
        # Store selected n-grams
        self.ngram_counts = dict(selected_ngrams)
        
        # Print information about selected n-grams
        print(f"\nSelected {len(selected_ngrams)} n-grams:")
        if selected_ngrams:
            min_freq = selected_ngrams[-1][1] / total_positions * 100
            max_freq = selected_ngrams[0][1] / total_positions * 100
            print(f"Frequency range: {min_freq:.3f}% to {max_freq:.3f}%")
        
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
            chunk_path = f"{base_path}_chunk{chunk_idx}.pt"
            torch.save(chunk_activations, chunk_path)
            print(f"Saved chunk {chunk_idx} with shape {chunk_activations.shape}")
            
            del chunk_activations
            torch.cuda.empty_cache()

    def load_activations_chunk(self, chunk_idx: int) -> Optional[torch.Tensor]:
        """Load a specific chunk of activations."""
        base_path = self.get_activations_cache_path()
        chunk_path = f"{base_path}_chunk{chunk_idx}.pt"
        
        if os.path.exists(chunk_path):
            return torch.load(chunk_path)
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
        # Initialize arrays to store all true labels and predictions
        all_true_labels = []
        all_pred_probs = []
        
        # Process each batch's predictions with memory optimization
        for batch_logits, batch_labels in zip(chunk_logits, chunk_labels):
            probs = torch.sigmoid(batch_logits)
            
            # Process in smaller slices to reduce memory usage
            slice_size = 100  # Reduced slice size for better memory management
            for start_idx in range(0, batch_labels.shape[0], slice_size):
                end_idx = min(start_idx + slice_size, batch_labels.shape[0])
                
                # Convert to float32 and move to CPU in smaller chunks
                true_labels = batch_labels[start_idx:end_idx].to(dtype=torch.float32).cpu().numpy()
                pred_probs = probs[start_idx:end_idx].to(dtype=torch.float32).cpu().numpy()
                
                # Append to lists
                all_true_labels.append(true_labels)
                all_pred_probs.append(pred_probs)
                
                del true_labels, pred_probs
                torch.cuda.empty_cache()
            
            del probs
            torch.cuda.empty_cache()
        
        # Concatenate all batches
        all_true_labels = np.concatenate(all_true_labels, axis=0)
        all_pred_probs = np.concatenate(all_pred_probs, axis=0)
        
        # Compute AUROC scores for all n-grams at once
        auroc_scores = {}
        for ngram in ngrams:
            idx = ngram_to_idx[ngram]
            if idx >= all_pred_probs.shape[-1]:
                auroc_scores[ngram] = 0.5
                continue
            
            try:
                # Extract the specific n-gram's predictions and labels
                true_labels = all_true_labels[..., idx].flatten()
                pred_probs = all_pred_probs[..., idx].flatten()
                
                # Compute AUROC
                auroc = roc_auc_score(true_labels, pred_probs)
                auroc_scores[ngram] = auroc
            except ValueError:
                auroc_scores[ngram] = 0.5
        
        # Clear memory
        del all_true_labels, all_pred_probs
        torch.cuda.empty_cache()
        
        return auroc_scores

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
        """Train both types of probes using chunked activations."""
        print("\nStarting probe training:")
        print(f"Number of n-grams selected: {len(ngrams)}")
        print(f"Frequency range: {min(self.ngram_counts.values()) / (len(tokens) * self.config.ctx_len) * 100:.3f}% to {max(self.ngram_counts.values()) / (len(tokens) * self.config.ctx_len) * 100:.3f}%")
        
        total_tokens = len(tokens) * self.config.ctx_len
        required_tokens = self.config.num_train_tokens + self.config.num_val_tokens
        if required_tokens > total_tokens:
            raise ValueError(f"Not enough tokens available ({total_tokens} < {required_tokens})")
            
        # Calculate total number of possible n-gram positions
        total_positions = total_tokens - (self.config.ngram_size - 1) * len(tokens)
        
        # Create frequency vs rank plot
        print("\nCreating frequency vs rank plot...")
        plt.figure(figsize=(12, 6))
        
        # Get all counts and sort them in descending order
        counts = sorted(self.ngram_counts.values(), reverse=True)
        if hasattr(self, 'ignored_ngrams'):
            ignored_counts = sorted(self.ignored_ngrams.values(), reverse=True)
            all_counts = ignored_counts + counts
            frequencies = [count / total_positions * 100 for count in all_counts]
            ranks = list(range(1, len(frequencies) + 1))
            
            # Plot ignored n-grams in red
            plt.plot(ranks[:len(ignored_counts)], frequencies[:len(ignored_counts)], 
                    'r-', alpha=0.7, linewidth=1, label='Ignored n-grams')
            # Plot kept n-grams in blue
            plt.plot(ranks[len(ignored_counts):], frequencies[len(ignored_counts):], 
                    'b-', alpha=0.7, linewidth=1, label='Kept n-grams')
        else:
            frequencies = [count / total_positions * 100 for count in counts]
            ranks = list(range(1, len(frequencies) + 1))
            plt.plot(ranks, frequencies, 'b-', alpha=0.7, linewidth=1, label='All n-grams')
        
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.title(f"N-gram Frequency Distribution (n={self.config.ngram_size})")
        plt.xlabel("Rank (log scale)")
        plt.ylabel("Frequency % (log scale)")
        
        # Add vertical lines for cutoffs
        if self.config.ignore_top_n > 0:
            plt.axvline(x=self.config.ignore_top_n, color='r', linestyle='--', alpha=0.5, 
                       label=f'Ignore top {self.config.ignore_top_n}')
        plt.axvline(x=self.config.top_m_ngrams + self.config.ignore_top_n, color='g', 
                   linestyle='--', alpha=0.5, 
                   label=f'Keep top {self.config.top_m_ngrams}')
        
        plt.legend()
        
        # Save plot
        os.makedirs(os.path.join(self.config.output_dir, "plots"), exist_ok=True)
        plot_path = os.path.join(self.config.output_dir, "plots", "ngram_frequency_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved frequency distribution plot to {plot_path}")
        
        # Print summary statistics
        print(f"\nTraining on {len(ngrams)} n-grams")
        print(f"Most frequent included n-gram appears in {frequencies[self.config.ignore_top_n]:.3f}% of positions")
        print(f"Least frequent included n-gram appears in {frequencies[self.config.ignore_top_n + len(ngrams) - 1]:.3f}% of positions")
        
        # Calculate number of chunks needed for training and validation
        num_train_chunks = math.ceil(self.config.num_train_sequences / self.config.chunk_size)
        num_val_chunks = math.ceil(self.config.num_val_sequences / self.config.chunk_size)
        
        results = {}
        
        for probe_type in ['first', 'final']:
            probe = NgramProbe(
                d_model=self.model.cfg.d_model,
                ngrams=ngrams,
                device=self.config.device,
                probe_type=probe_type,
                dtype=self.config.dtype
            ).to(self.config.device)
            
            optimizer = torch.optim.Adam(probe.parameters(), lr=self.config.learning_rate)
            
            # Training loop with progress bar
            pbar = tqdm(range(num_train_chunks), desc=f"Training {probe_type}-position probe")
            running_loss = 0
            num_batches = 0
            
            for chunk_idx in pbar:
                activations = self.load_activations_chunk(chunk_idx)
                if activations is None:
                    continue
                
                chunk_tokens = tokens[chunk_idx * self.config.chunk_size:(chunk_idx + 1) * self.config.chunk_size]
                
                for i in range(0, len(chunk_tokens), self.config.batch_size):
                    batch_end = min(i + self.config.batch_size, len(chunk_tokens))
                    batch_tokens = chunk_tokens[i:batch_end]
                    batch_activations = activations[i:batch_end]
                    
                    activations_batch, labels = self.prepare_probe_data(
                        batch_tokens, batch_activations, ngrams, ngram_to_idx, probe_type
                    )
                    
                    logits = probe(activations_batch)
                    loss = probe.compute_loss(logits, labels, self.config.positive_weight)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    running_loss = 0.9 * running_loss + 0.1 * loss.item() if num_batches > 0 else loss.item()
                    num_batches += 1
                    pbar.set_postfix({'loss': f'{running_loss:.4f}'})
                    
                    del activations_batch, labels, logits
                    torch.cuda.empty_cache()
                
                del activations
                torch.cuda.empty_cache()
            
            # Evaluation
            probe.eval()
            
            print(f"\nEvaluating {probe_type}-position probe...")
            # Initialize accumulators for AUROC scores
            auroc_accumulator = {ngram: [] for ngram in ngrams}
            
            for chunk_idx in range(num_train_chunks, num_train_chunks + num_val_chunks):
                activations = self.load_activations_chunk(chunk_idx)
                if activations is None:
                    continue
                    
                chunk_tokens = tokens[chunk_idx * self.config.chunk_size:(chunk_idx + 1) * self.config.chunk_size]
                chunk_logits = []
                chunk_labels = []
                
                with torch.no_grad():
                    for i in tqdm(range(0, len(chunk_tokens), self.config.batch_size)):
                        batch_end = min(i + self.config.batch_size, len(chunk_tokens))
                        batch_tokens = chunk_tokens[i:batch_end]
                        batch_activations = activations[i:batch_end]
                        
                        activations_batch, labels = self.prepare_probe_data(
                            batch_tokens, batch_activations, ngrams, ngram_to_idx, probe_type
                        )
                        logits = probe(activations_batch)
                        
                        chunk_logits.append(logits.cpu())
                        chunk_labels.append(labels.cpu())
                        
                        del activations_batch, labels, logits
                        torch.cuda.empty_cache()
                
                # Compute metrics for this chunk
                chunk_results = self.compute_chunk_metrics(probe, chunk_logits, chunk_labels, ngrams, ngram_to_idx)
                
                # Accumulate AUROC scores
                for ngram in ngrams:
                    auroc_accumulator[ngram].append(chunk_results[ngram])
                
                del activations, chunk_logits, chunk_labels, chunk_results
                torch.cuda.empty_cache()
            
            # Average AUROC scores across chunks
            results[probe_type] = {
                ngram: np.mean(auroc_accumulator[ngram]) 
                for ngram in ngrams
            }
            
            # Save final model
            final_model_path = os.path.join(self.config.output_dir, f"probe_{probe_type}_final.pt")
            probe.save(final_model_path)
        
        return results
    
    def save_results(self, results: Dict[str, Dict[str, float]], output_dir: str):
        """Save results and create comparison plots."""
        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save results JSON
        output_file = os.path.join(output_dir, "probe_results.json")
        serializable_results = {
            probe_type: {
                str(ngram): {
                    "auroc": stats,
                    "count": self.ngram_counts[ngram]
                } for ngram, stats in probe_results.items()
            } for probe_type, probe_results in results.items()
        }
        
        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        # Extract data for plotting
        first_log_errors = np.log10(1 - np.array(list(results['first'].values())))
        final_log_errors = np.log10(1 - np.array(list(results['final'].values())))
        
        # Calculate normalized frequencies
        total_tokens = len(self.tokens) * self.config.ctx_len
        counts = np.array([self.ngram_counts[ngram] for ngram in results['first'].keys()])
        frequencies = counts / total_tokens
        log_frequencies = np.log10(frequencies)
        
        # Create histogram with consistent bins
        plt.figure(figsize=(12, 6))
        
        # Calculate common bins based on all log error rates
        all_log_errors = np.concatenate([first_log_errors, final_log_errors])
        bins = np.linspace(min(all_log_errors), max(all_log_errors), self.config.histogram_bins)
        
        plt.hist(first_log_errors, bins=bins, alpha=0.5, label='First Position')
        plt.hist(final_log_errors, bins=bins, alpha=0.5, label='Final Position')
        plt.title("Distribution of Log10 Error Rates (log10(1 - AUROC)) by Probe Type")
        plt.xlabel("Log10 Error Rate")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "log_error_rate_histogram_comparison.png"))
        plt.close()
        
        # Create scatter plot with normalized frequencies
        plt.figure(figsize=(12, 6))
        plt.scatter(log_frequencies, first_log_errors, alpha=0.5, label='First Position')
        plt.scatter(log_frequencies, final_log_errors, alpha=0.5, label='Final Position')
        plt.title("Log10 Error Rate vs Log10 Frequency by Probe Type")
        plt.xlabel("Log10 Frequency (log10(occurrences / total tokens))")
        plt.ylabel("Log10 Error Rate (log10(1 - AUROC))")
        plt.legend()
        
        # Add trend lines with R² values
        for log_errors, style, label in [(first_log_errors, 'r--', 'First Position'), 
                                       (final_log_errors, 'b--', 'Final Position')]:
            # Linear fit in log-log space
            mask = ~np.isnan(log_frequencies) & ~np.isnan(log_errors) & ~np.isinf(log_frequencies) & ~np.isinf(log_errors)
            z = np.polyfit(log_frequencies[mask], log_errors[mask], 1)
            p = np.poly1d(z)
            r2 = np.corrcoef(log_frequencies[mask], log_errors[mask])[0,1]**2
            
            # Plot trend line
            x_range = np.linspace(log_frequencies.min(), log_frequencies.max(), 100)
            plt.plot(x_range, p(x_range), style, alpha=0.8, label=f'{label} (R² = {r2:.3f})')
        
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "log_error_rate_vs_freq_comparison.png"))
        plt.close()
        
        # Create direct comparison scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(first_log_errors, final_log_errors, alpha=0.5)
        plt.plot([min(all_log_errors), max(all_log_errors)], 
                [min(all_log_errors), max(all_log_errors)], 'r--', alpha=0.8)
        plt.title("First Position vs Final Position Log10 Error Rates")
        plt.xlabel("First Position Log10 Error Rate (log10(1 - AUROC))")
        plt.ylabel("Final Position Log10 Error Rate (log10(1 - AUROC))")
        
        correlation = np.corrcoef(first_log_errors, final_log_errors)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                 transform=plt.gca().transAxes)
        
        plt.savefig(os.path.join(plots_dir, "first_vs_final_log_error_rate.png"))
        plt.close()

def main():
    config = NgramProbingConfig()
    
    try:
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "cache"), exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        return
    
    # Run for different n-gram sizes
    for n in range(1, 10):
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
        os.makedirs(os.path.join(current_config.output_dir, "cache"), exist_ok=True)
        
        prober = NgramProber(current_config)
        dataset = prober.load_dataset()
        tokens = prober.get_tokens(dataset)
        
        ngrams = prober.get_ngrams(tokens)
        ngram_list = list(ngrams.keys())
        ngram_to_idx = {ngram: idx for idx, ngram in enumerate(ngram_list)}
        
        # Print information about ignored n-grams
        if hasattr(prober, 'ignored_ngrams') and current_config.ignore_top_n > 0:
            total_tokens = len(tokens) * current_config.ctx_len
            print(f"\nIgnored {current_config.ignore_top_n} most frequent n-grams:")
            for ngram, count in prober.ignored_ngrams.items():
                freq = count / total_tokens * 100
                print(f"N-gram: {prober.model.to_string(torch.tensor(ngram))}, "
                      f"frequency: {freq:.3f}%")
        
        # Generate or load activations
        activations = prober.load_activations_chunk(0)
        if activations is None:
            prober.generate_and_save_activations(tokens)
            activations = prober.load_activations_chunk(0)
        
        results = prober.train_probe(tokens, ngram_list, ngram_to_idx)
        prober.save_results(results, current_config.output_dir)
        
        print(f"\nCompleted n-gram size {n}")
        print(f"Results saved to {current_config.output_dir}")
        
        # Clear memory
        del prober, dataset, tokens, ngrams, ngram_list, ngram_to_idx, activations, results
        torch.cuda.empty_cache()
    
    print("\nCompleted all n-gram sizes!")

if __name__ == "__main__":
    main() 