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
            
        all_ngrams = []
        for batch_start in tqdm(range(0, len(tokens), self.config.model_batch_size), desc="Finding n-grams"):
            batch = tokens[batch_start:batch_start + self.config.model_batch_size].cpu().numpy()
            
            for seq in batch:
                seq_ngrams = [
                    tuple(seq[j:j + self.config.ngram_size].tolist())
                    for j in range(len(seq) - self.config.ngram_size + 1)
                ]
                all_ngrams.extend(seq_ngrams)
            
            del batch
            if batch_start % (5 * self.config.model_batch_size) == 0:
                torch.cuda.empty_cache()
        
        ngram_counts = Counter(all_ngrams)
        
        # Get top N+k n-grams, then remove top k
        top_ngrams = ngram_counts.most_common(self.config.top_m_ngrams + self.config.ignore_top_n)[self.config.ignore_top_n:]
        self.ngram_counts = dict(top_ngrams)
        
        # Store ignored n-grams for later use
        if self.config.ignore_top_n > 0:
            self.ignored_ngrams = dict(ngram_counts.most_common(self.config.ignore_top_n))
        
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
                     f"ctx{self.config.ctx_len}.pt")
        return os.path.join(self.config.output_dir, "cache", cache_name)

    def generate_and_save_activations(self, tokens: torch.Tensor) -> None:
        """Generate activations for all tokens and save them to disk."""
        cache_path = self.get_activations_cache_path()
        
        print("\nGenerating activations...")
        all_activations = []
        
        for i in tqdm(range(0, len(tokens), self.config.model_batch_size)):
            batch_tokens = tokens[i:i + self.config.model_batch_size]
            with torch.no_grad():
                activations = self.get_activations_batch(batch_tokens)
                all_activations.append(activations.cpu())
            
            del activations
            torch.cuda.empty_cache()
        
        print("Concatenating activations...")
        activations = torch.cat(all_activations, dim=0)
        print(f"Total activations shape: {activations.shape}")
        
        print("Saving activations to disk...")
        torch.save(activations, cache_path)
        print(f"Saved activations to {cache_path}")
        
        del activations, all_activations
        torch.cuda.empty_cache()

    def load_activations(self) -> Optional[torch.Tensor]:
        """Load cached activations if they exist."""
        cache_path = self.get_activations_cache_path()
        if os.path.exists(cache_path):
            print("\nLoading cached activations...")
            return torch.load(cache_path)
        return None

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
        activations: torch.Tensor,
        ngrams: List[Tuple[int, ...]],
        ngram_to_idx: Dict[Tuple[int, ...], int]
    ) -> Dict[str, Dict[str, float]]:
        """Train both types of probes using pre-computed activations."""
        total_tokens = len(tokens) * self.config.ctx_len
        required_tokens = self.config.num_train_tokens + self.config.num_val_tokens
        if required_tokens > total_tokens:
            raise ValueError(f"Not enough tokens available ({total_tokens} < {required_tokens})")
            
        # Calculate total number of possible n-gram positions
        total_positions = total_tokens - (self.config.ngram_size - 1) * len(tokens)
        
        # Print n-gram frequency information
        print("\nN-gram frequency statistics:")
        counts = list(self.ngram_counts.values())
        max_freq = max(counts) / total_positions * 100
        min_freq = min(counts) / total_positions * 100
        print(f"Training on {len(ngrams)} n-grams:")
        print(f"Most frequent included n-gram appears in {max_freq:.3f}% of positions")
        print(f"Least frequent included n-gram appears in {min_freq:.3f}% of positions")
        
        # Print information about ignored n-grams
        if hasattr(self, 'ignored_ngrams') and self.config.ignore_top_n > 0:
            print(f"\nIgnored {self.config.ignore_top_n} most frequent n-grams:")
            for ngram, count in self.ignored_ngrams.items():
                freq = count / total_tokens * 100
                print(f"N-gram: {self.model.to_string(torch.tensor(ngram))}, "
                      f"frequency: {freq:.3f}%")
        
        train_tokens = tokens[:self.config.num_train_sequences]
        val_tokens = tokens[self.config.num_train_sequences:self.config.num_train_sequences + self.config.num_val_sequences]
        train_activations = activations[:self.config.num_train_sequences]
        val_activations = activations[self.config.num_train_sequences:self.config.num_train_sequences + self.config.num_val_sequences]
        
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
            pbar = tqdm(range(0, len(train_tokens), self.config.batch_size), 
                       desc=f"Training {probe_type}-position probe")
            running_loss = 0
            num_batches = 0
            
            for i in pbar:
                batch_end = min(i + self.config.batch_size, len(train_tokens))
                batch_tokens = train_tokens[i:batch_end]
                batch_activations = train_activations[i:batch_end]
                
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
            
            # Evaluation
            probe.eval()
            with torch.no_grad():
                all_logits = []
                all_labels = []
                
                for i in tqdm(range(0, len(val_tokens), self.config.batch_size), 
                            desc=f"Evaluating {probe_type}-position probe"):
                    batch_end = min(i + self.config.batch_size, len(val_tokens))
                    batch_tokens = val_tokens[i:batch_end]
                    batch_activations = val_activations[i:batch_end]
                    
                    activations_batch, labels = self.prepare_probe_data(
                        batch_tokens, batch_activations, ngrams, ngram_to_idx, probe_type
                    )
                    logits = probe(activations_batch)
                    
                    all_logits.append(logits.cpu())
                    all_labels.append(labels.cpu())
                    
                    del activations_batch, labels, logits
                    torch.cuda.empty_cache()
                
                logits = torch.cat(all_logits, dim=0)
                labels = torch.cat(all_labels, dim=0)
                aurocs = probe.compute_metrics(logits, labels)
                
                del logits, labels, all_logits, all_labels
                torch.cuda.empty_cache()
            
            # Save final model
            final_model_path = os.path.join(self.config.output_dir, f"probe_{probe_type}_final.pt")
            probe.save(final_model_path)
            results[probe_type] = aurocs
        
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
    
    prober = NgramProber(config)
    dataset = prober.load_dataset()
    tokens = prober.get_tokens(dataset)
    
    ngrams = prober.get_ngrams(tokens)
    ngram_list = list(ngrams.keys())
    ngram_to_idx = {ngram: idx for idx, ngram in enumerate(ngram_list)}
    
    # Print information about ignored n-grams
    if hasattr(prober, 'ignored_ngrams') and config.ignore_top_n > 0:
        total_tokens = len(tokens) * config.ctx_len
        print(f"\nIgnored {config.ignore_top_n} most frequent n-grams:")
        for ngram, count in prober.ignored_ngrams.items():
            freq = count / total_tokens * 100
            print(f"N-gram: {prober.model.to_string(torch.tensor(ngram))}, "
                  f"frequency: {freq:.3f}%")
    
    # Generate or load activations
    activations = prober.load_activations()
    if activations is None:
        prober.generate_and_save_activations(tokens)
        activations = prober.load_activations()
    
    results = prober.train_probe(tokens, activations, ngram_list, ngram_to_idx)
    prober.save_results(results, config.output_dir)

if __name__ == "__main__":
    main() 