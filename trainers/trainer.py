import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from utils.io import save_json

class EEGChannelARDataset(Dataset):
    """
    Autoregressive training dataset.(Normalization is applied dynamically in __getitem__ only; do NOT normalize the data elsewhere.)
    """
    def __init__(self, data: np.ndarray, scheduler, selector, norm_stats: dict):
        self.data = data.astype(np.float32)  
        self.norm_stats = norm_stats       
        self.N, self.C_total, self.L = self.data.shape
        self.scheduler = scheduler
        self.selector = selector
        self.num_steps = scheduler.num_steps
        self.max_in_channels = scheduler.target_channels - min(scheduler.step_sizes)
        
        # New: obtain initial channel indices
        self.initial_channels = scheduler.initial_channels

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.data[idx]  # (C, L)

        # Dynamic normalization (consistent with SR_VAE, using either global or per-channel statistics)
        if self.norm_stats['mode'] == 'global':
            x = (x - self.norm_stats['mean']) / (self.norm_stats['std'] + 1e-8)
        elif self.norm_stats['mode'] == 'per_channel':
            mean = np.array(self.norm_stats['channel_means'], dtype=np.float32).reshape(-1, 1)
            std = np.array(self.norm_stats['channel_stds'], dtype=np.float32).reshape(-1, 1)
            x = (x - mean) / (std + 1e-8)

        inputs = np.zeros((self.num_steps, self.max_in_channels, self.L), dtype=np.float32)
        targets = np.zeros((self.num_steps, max(self.scheduler.step_sizes), self.L), dtype=np.float32)

        for s in range(self.num_steps):
            known_idx = self.scheduler.get_known_channels(s)  
            out_idx = self.scheduler.get_step_out_indices(s)
            step_size = len(out_idx)
            
            # Teacher forcing: use ground truth for known channels
            inputs[s, :len(known_idx), :] = x[known_idx, :]  
            targets[s, :step_size, :] = x[out_idx, :]

        return torch.from_numpy(inputs), torch.from_numpy(targets)


class Trainer:
    """Unified training controller."""
    def __init__(self, model, config, device, run_dir):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.run_dir = run_dir
        
        # Load training configuration
        lr = config['train']['lr']
        weight_decay = config['train'].get('weight_decay', 0.0)
        model_name = config['model']['name']
        
        # Select optimizer based on configuration or model type
        optimizer_type = config['train'].get('optimizer', None)
        
        if optimizer_type is None:
            # Automatically choose optimizer if not explicitly specified
            if model_name == 'Transformer':
                optimizer_type = 'adamw'
            else:
                optimizer_type = 'adam'
        
        # Create optimizer
        if optimizer_type.lower() == 'adamw':
            print(f"Using AdamW optimizer (lr={lr}, weight_decay={weight_decay})")
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adam':
            print(f"Using Adam optimizer (lr={lr}, weight_decay={weight_decay})")
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        self.early_stop_patience = config['train']['early_stopping']['patience']
        self.early_stop_min_delta = config['train']['early_stopping']['min_delta']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = -1
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for inputs, targets in dataloader:
            B, S, C_in, L = inputs.shape
            _, _, step_max, _ = targets.shape
            
            inputs = inputs.view(B * S, C_in, L).to(self.device)
            targets = targets.view(B * S, step_max, L).to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss only on valid (non-padded) target channels
            loss = F.mse_loss(outputs, targets[:, :outputs.size(1), :])
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * B
            total_samples += B
        
        return total_loss / max(total_samples, 1)
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        for inputs, targets in dataloader:
            B, S, C_in, L = inputs.shape
            _, _, step_max, _ = targets.shape
            
            inputs = inputs.view(B * S, C_in, L).to(self.device)
            targets = targets.view(B * S, step_max, L).to(self.device)
            
            outputs = self.model(inputs)
            loss = F.mse_loss(outputs, targets[:, :outputs.size(1), :])
            
            total_loss += loss.item() * B
            total_samples += B
        
        return total_loss / max(total_samples, 1)
    
    def fit(self, train_loader, val_loader, epochs):
        log_path = os.path.join(self.run_dir, "logs", "train_log.jsonl")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            # Save training log
            log_entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            with open(log_path, "a") as f:
                f.write(str(log_entry) + "\n")
            
            # Save checkpoint
            ckpt = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            }
            torch.save(ckpt, os.path.join(self.run_dir, "checkpoints", "last_checkpoint.pth"))
            
            # Early stopping and best model tracking
            if val_loss < self.best_val_loss - self.early_stop_min_delta:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                torch.save(ckpt, os.path.join(self.run_dir, "checkpoints", "best_model.pth"))
            else:
                self.patience_counter += 1
            
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                  f"(best {self.best_val_loss:.6f} @ {self.best_epoch})")
            
            if self.patience_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break