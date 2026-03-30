
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import torch
from torch.nn.parallel import DataParallel

from utils.io import load_json, save_json
from data.adapters import load_data
from models.registry import get_model
from utils.norm import normalize, denormalize


def load_checkpoint_smart(model, checkpoint_path, device, verbose=True):
    """
    Smart checkpoint loader that automatically handles the 'module.' prefix introduced by DataParallel.
    
    Supported cases:
    1. Checkpoint has 'module.' and current model uses DataParallel
    2. Checkpoint has 'module.' and current model does NOT use DataParallel
    3. Checkpoint has NO 'module.' and current model uses DataParallel
    4. Checkpoint has NO 'module.' and current model does NOT use DataParallel
    
    Args:
        model: Current model (may be wrapped by DataParallel)
        checkpoint_path: Path to checkpoint file
        device: Device
        verbose: Whether to print detailed information
    
    Returns:
        checkpoint: Full checkpoint dictionary
    """
    if verbose:
        print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Analyze checkpoint
    checkpoint_keys = list(state_dict.keys())
    checkpoint_has_module = checkpoint_keys[0].startswith('module.')
    
    # Analyze current model
    model_is_parallel = isinstance(model, DataParallel)
    target_model = model.module if model_is_parallel else model
    target_keys = list(target_model.state_dict().keys())
    
    if verbose:
        print(f"  Checkpoint info:")
        print(f"    - Has 'module.' prefix: {checkpoint_has_module}")
        print(f"    - Sample keys: {checkpoint_keys[:2]}")
        print(f"  Current model info:")
        print(f"    - Is DataParallel: {model_is_parallel}")
        print(f"    - Sample keys: {target_keys[:2]}")
    
    # Try direct loading
    try:
        if model_is_parallel:
            model.module.load_state_dict(state_dict, strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
        
        if verbose:
            print(f"  ✓ Loaded directly (no conversion needed)")
        return checkpoint
    
    except RuntimeError:
        # Prefix conversion required
        if checkpoint_has_module and not model_is_parallel:
            # Remove 'module.' prefix
            if verbose:
                print(f"  🔧 Converting: Removing 'module.' prefix")
            
            new_state_dict = {
                k.replace('module.', '', 1): v 
                for k, v in state_dict.items()
            }
            model.load_state_dict(new_state_dict, strict=True)
            
            if verbose:
                print(f"  ✓ Loaded after removing 'module.' prefix")
        
        elif not checkpoint_has_module and model_is_parallel:
            # Add 'module.' prefix
            if verbose:
                print(f"  🔧 Converting: Adding 'module.' prefix")
            
            new_state_dict = {
                f'module.{k}': v 
                for k, v in state_dict.items()
            }
            model.load_state_dict(new_state_dict, strict=True)
            
            if verbose:
                print(f"  ✓ Loaded after adding 'module.' prefix")
        
        else:
            # Unsupported case
            print(f"\n❌ Error: Cannot load checkpoint!")
            print(f"  Checkpoint has 'module.': {checkpoint_has_module}")
            print(f"  Model is DataParallel: {model_is_parallel}")
            print(f"  Checkpoint keys: {checkpoint_keys[:3]}")
            print(f"  Model keys: {target_keys[:3]}")
            raise RuntimeError("Failed to load checkpoint with smart conversion")
    
    return checkpoint


def compute_nmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalized Mean Squared Error (NMSE)."""
    num = np.sum((pred - true) ** 2)
    den = np.sum(true ** 2) + 1e-8
    return float(num / den)


def compute_snr(pred: np.ndarray, true: np.ndarray) -> float:
    """Signal-to-Noise Ratio (SNR, dB)."""
    num = np.sum(true ** 2)
    den = np.sum((pred - true) ** 2) + 1e-8
    return float(10.0 * np.log10(num / den))


def compute_pcc(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Pearson Correlation Coefficient (PCC).
    The data is reshaped to (N*C, L). Correlation is computed per channel sequence and then averaged.
    """
    N, C, L = true.shape
    pred_flat = pred.reshape(N * C, L)
    true_flat = true.reshape(N * C, L)

    pred_mean = pred_flat.mean(axis=1, keepdims=True)
    true_mean = true_flat.mean(axis=1, keepdims=True)

    pred_c = pred_flat - pred_mean
    true_c = true_flat - true_mean

    num = np.sum(pred_c * true_c, axis=1)
    den = np.sqrt(
        np.sum(pred_c ** 2, axis=1) * np.sum(true_c ** 2, axis=1) + 1e-8
    )
    corr = num / den
    return float(corr.mean())
# ...existing code...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Run directory created during training ""(contains config.json, norm_stats.json, checkpoints/best_model.pth)"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # ==================== 1. Load configuration ====================
    config_path = os.path.join(args.run_dir, "config.json")
    norm_stats_path = os.path.join(args.run_dir, "norm_stats.json")
    ckpt_path = os.path.join(args.run_dir, "checkpoints", "best_model.pth")
    
    assert os.path.isfile(config_path), f"config.json not found at {config_path}"
    assert os.path.isfile(norm_stats_path), f"norm_stats.json not found at {norm_stats_path}"
    assert os.path.isfile(ckpt_path), f"best_model.pth not found at {ckpt_path}"

    config = load_json(config_path)
    norm_stats = load_json(norm_stats_path)

    # ==================== 2. Set device ====================
    device_str = config['runtime'].get('device')
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # ==================== 3. Load test data ====================
    _, test_data, _ = load_data(config)  # (N, C, L)
    N, C_total, L = test_data.shape
    print(f"Test data shape: {test_data.shape}")

    # ==================== 4. Extract configuration parameters ====================
    base_channels = config['sr']['base_channels']
    target_channels = config['sr']['target_channels']
    step_sizes = config['schedule']['step_sizes']
    
    # 🔧 Key fix: extract initial and target channel indices
    initial_channels = config['sr'].get('initial_channels', list(range(base_channels)))
    target_channel_indices = config['sr'].get('target_channels_indices', list(range(target_channels)))
    
    print(f"\n📌 Initial channels (indices): {initial_channels}")
    print(f"   Initial channel count: {len(initial_channels)}")
    print(f"📌 Target channels (indices): {target_channel_indices}")
    print(f"   Target channel count: {len(target_channel_indices)}")
    
    # Validate initial channel count
    assert len(initial_channels) == base_channels, \
        f"initial_channels length ({len(initial_channels)}) != base_channels ({base_channels})"
    
    # Validate target channel count
    assert len(target_channel_indices) == target_channels, \
        f"target_channel_indices length ({len(target_channel_indices)}) != target_channels ({target_channels})"
    
    # Compute channels to be generated
    initial_set = set(initial_channels)
    generated_channel_indices = [ch for ch in target_channel_indices if ch not in initial_set]
    
    print(f"📌 Generated channels (indices): {generated_channel_indices}")
    print(f"   Generated channel count: {len(generated_channel_indices)}")
    
    # Validate step size sum
    expected_total = len(generated_channel_indices)
    actual_total = sum(step_sizes)
    assert actual_total == expected_total, \
        f"step_sizes sum={actual_total} != expected generated channels={expected_total}"
    
    # Compute model input/output dimensions (consistent with training)
    max_in_channels = target_channels - min(step_sizes)
    step_channels = max(step_sizes)  
    
    print(f"\nBase channels: {base_channels}")
    print(f"Target channels: {target_channels}")
    print(f"Step sizes: {step_sizes}")
    print(f"Max input channels: {max_in_channels}")
    print(f"Step channels (output): {step_channels}")

    # ==================== 5. Build model and load weights ====================
    model = get_model(
        config,
        in_channels=max_in_channels,
        out_channels=step_channels
    ).to(device)
    
    ckpt = load_checkpoint_smart(model, ckpt_path, device, verbose=True)
    
    model.eval()
    print(f"✓ Model loaded successfully (epoch {ckpt.get('epoch', 'N/A')})")

    # ==================== 6. Normalize test data ====================
    test_norm = normalize(test_data.astype(np.float32), norm_stats)  # (N, C, L)
    test_norm_t = torch.from_numpy(test_norm).to(device)
    print(f"Data normalized using mode: {norm_stats['mode']}")

    # ==================== 7. Initialize reconstruction (normalized domain) ====================
    recon_norm = torch.zeros(N, C_total, L, device=device, dtype=test_norm_t.dtype)
    
    # 🔧 Key fix: copy initial channels to original positions
    for orig_idx in initial_channels:
        recon_norm[:, orig_idx, :] = test_norm_t[:, orig_idx, :]
    
    print(f"✓ Initialized with channels at original positions: {initial_channels}")

    # ==================== 8. Autoregressive generation ====================
    num_steps = len(step_sizes)
    print(f"\nStarting autoregressive generation ({num_steps} steps)...")

    with torch.no_grad():
        current_channels = list(initial_channels)
        gen_idx = 0  
        
        for s in range(num_steps):
            step_size = step_sizes[s]  
            
            step_gen_indices = generated_channel_indices[gen_idx:gen_idx + step_size]
            
            print(f"Step {s+1}/{num_steps}: Generating {step_size} channels at positions {step_gen_indices}")
            print(f"  Current known channels: {len(current_channels)} channels")
            
            for start_idx in range(0, N, args.batch_size):
                end_idx = min(N, start_idx + args.batch_size)
                batch_size = end_idx - start_idx

                inp = torch.zeros(
                    batch_size,
                    max_in_channels,
                    L,
                    device=device,
                    dtype=test_norm_t.dtype
                )
                
                # 🔧 Key fix: gather data from known channel positions
                for i, ch_idx in enumerate(current_channels):
                    inp[:, i, :] = recon_norm[start_idx:end_idx, ch_idx, :]

                out = model(inp)  # (batch_size, step_channels, L)
                
                # 🔧 Key fix: write predictions back to original positions
                for i, ch_idx in enumerate(step_gen_indices):
                    recon_norm[start_idx:end_idx, ch_idx, :] = out[:, i, :]

            current_channels.extend(step_gen_indices)
            gen_idx += step_size
            
            print(f"  ✓ Generated channels at positions {step_gen_indices}")
            print(f"  Total known channels now: {len(current_channels)}")

    print("\nAutoregressive generation completed!")

    # ==================== 9. Denormalization ====================
    recon = denormalize(recon_norm.cpu().numpy(), norm_stats)
    
    # 🔧 Key fix: replace initial channels with ground truth at original positions
    for orig_idx in initial_channels:
        recon[:, orig_idx, :] = test_data[:, orig_idx, :]
    
    print("Reconstruction denormalized and initial channels replaced with ground truth at original positions.")

    # ==================== 10. Compute metrics ====================
    print("\n" + "="*50)
    print("Computing metrics...")
    
    recon_target = recon[:, target_channel_indices, :]
    test_target = test_data[:, target_channel_indices, :]
    
    nmse_all = compute_nmse(recon_target, test_target)
    snr_all = compute_snr(recon_target, test_target)
    pcc_all = compute_pcc(recon_target, test_target)

    recon_gen = recon[:, generated_channel_indices, :]
    test_gen = test_data[:, generated_channel_indices, :]
    
    nmse_gen = compute_nmse(recon_gen, test_gen)
    snr_gen = compute_snr(recon_gen, test_gen)
    pcc_gen = compute_pcc(recon_gen, test_gen)

    # ==================== 11. Print results ====================
    print("\n=== Test Metrics (All Target Channels) ===")
    print(f"NMSE_all: {nmse_all:.6e}")
    print(f"SNR_all:  {snr_all:.3f} dB")
    print(f"PCC_all:  {pcc_all:.6f}")

    print("\n=== Test Metrics (Generated Channels Only) ===")
    print(f"NMSE_gen: {nmse_gen:.6e}")
    print(f"SNR_gen:  {snr_gen:.3f} dB")
    print(f"PCC_gen:  {pcc_gen:.6f}")
    print("="*50)

    # ==================== 12. Save results ====================
    test_dir = os.path.join(args.run_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    recon_path = os.path.join(test_dir, "reconstructed_test_data.npy")
    np.save(recon_path, recon)

    metrics = {
        "nmse_all": nmse_all,
        "snr_all": snr_all,
        "pcc_all": pcc_all,
        "nmse_gen": nmse_gen,
        "snr_gen": snr_gen,
        "pcc_gen": pcc_gen,
        "base_channels": base_channels,
        "target_channels": target_channels,
        "step_sizes": step_sizes,
        "max_in_channels": max_in_channels,
        "step_channels": step_channels,
        "initial_channels": initial_channels,
        "target_channel_indices": target_channel_indices,
        "generated_channel_indices": generated_channel_indices
    }
    metrics_path = os.path.join(test_dir, "test_metrics.json")
    save_json(metrics, metrics_path)

    channels_info = {
        "initial_channels": initial_channels, 
        "target_channel_indices": target_channel_indices,  
        "generated_channel_indices": generated_channel_indices, 
        "generation_order": [],  
    }
    
    gen_idx = 0
    for step_size in step_sizes:
        step_channels = generated_channel_indices[gen_idx:gen_idx + step_size]
        channels_info["generation_order"].append(step_channels)
        gen_idx += step_size
    
    channels_path = os.path.join(test_dir, "generated_channels.json")
    save_json(channels_info, channels_path)

    # ==================== 13. Print saved paths ====================
    print(f"\n✓ Reconstructed data saved to: {recon_path}")
    print(f"  Shape: {recon.shape} (keeps original channel dimension)")
    print(f"✓ Metrics saved to: {metrics_path}")
    print(f"✓ Channel info saved to: {channels_path}")
    print(f"\nAll results saved in: {test_dir}")




if __name__ == "__main__":
    main()