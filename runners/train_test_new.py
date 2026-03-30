import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils.io import create_run_dir, save_json
from utils.seed import set_seed
from utils.norm import compute_norm_stats
from utils.config_builder import build_seed_config, list_seed_configs
from data.adapters import load_data
from schedulers.step_scheduler import StepScheduler
from channel_selectors.strategies import get_selector
from models.registry import get_model
from trainers.trainer import Trainer, EEGChannelARDataset

torch.backends.cudnn.enabled = False
torch.backends.mkldnn.enabled = False


def setup_parser():
    """
    Setup command-line argument parser
    """
    parser = argparse.ArgumentParser(
        description="SEED Dataset EEG Channel Super-Resolution Training & Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python runners/train_test_new.py --list_configs
  python runners/train_test_new.py --device cuda:2
  
  # Custom selector and step sizes
  python runners/train_test_new.py --selector geometric --step_sizes 10,10,10
  python runners/train_test_new.py --selector matrix --step_sizes 5,5,5,5,5,5
  
  # Specify initial channels
  python runners/train_test_new.py --initial_channels 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,1
  
  # Complete example
  python runners/train_test_new.py \
    --model Conv_default \
    --sr_ratio half \
    --selector geometric \
    --step_sizes 5,10,15 \
    --device cuda:2 \
    --epochs 50
        """
    )
    
    # ==================== Main Configuration File ====================
    parser.add_argument(
        "--master_config",
        type=str,
        default="",
        help="Path to main configuration file"
    )
    
    # ==================== Data Source Selection ====================
    parser.add_argument(
        "--data_source",
        type=str,
        default="cross_subject",
        choices=["random_split", "cross_subject"],
        help="""Data source selection:
  random_split    - Random train/test split
  cross_subject   - Cross-subject split
Default: random_split"""
    )
    
    # ==================== Super-Resolution Ratio Selection ====================
    parser.add_argument(
        "--sr_ratio",
        type=str,
        default="eighth",
        choices=["half", "quarter", "eighth"],
        help="""Super-resolution ratio selection:
  half     - 32→62 channels (1/2 super-resolution)
  quarter  - 16→62 channels (1/4 super-resolution)
  eighth   - 8→62 channels  (1/8 super-resolution)
Default: half"""
    )
    
    # ==================== New: Selector Selection ====================
    parser.add_argument(
        "--selector",
        type=str,
        default="matrix",
        choices=["sequential", "random", "geometric", "matrix", "custom"],
        help="""Channel selector (overrides config file):
  sequential  - Sequential selection (by index: 32, 33, 34, ...)
  random      - Random selection (requires --selector_seed)
  geometric   - Geometric distance selection (based on electrode spatial positions, requires coordinate file)
  matrix      - Matrix distance selection (based on channel correlation matrix)
  custom      - Custom order (requires --custom_order)

Default: matrix"""
    )
    
    # ==================== New: Selector Mode ====================
    parser.add_argument(
        "--selector_mode",
        type=str,
        default="uniform",
        choices=["uniform", "greedy", "balanced"],
        help="""Selector mode (only for geometric and matrix):
  uniform   - Uniform coverage
  greedy    - Greedy selection
  balanced  - Balanced mode (matrix only)
Default: greedy"""
    )
    
    # ==================== New: Electrode Coordinates File ====================
    parser.add_argument(
        "--electrode_positions",
        type=str,
        default="configs/electrodes/SEED_62ch.json",
        help="Electrode coordinates file path (only for geometric selector)"
    )
    
    # ==================== New: Random Seed (Selector Specific) ====================
    parser.add_argument(
        "--selector_seed",
        type=int,
        default=42,
        help="Random seed for selector (only for random selector)"
    )
    
    # ==================== New: Custom Channel Order ====================
    parser.add_argument(
        "--custom_order",
        type=str,
        default=None,
        help="""Custom channel generation order (only for custom selector):
Format: comma-separated channel indices
Example: --custom_order 32,33,34,...,61"""
    )
    
    # ==================== New: Step Sizes Array ====================
    parser.add_argument(
        "--step_sizes",
        type=str,
        default="27,27",
        help="""Step sizes array (overrides config file):
Format: comma-separated integers
Example: --step_sizes 10,10,10 or --step_sizes 5,5,5,5,5,5

1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1

15-16-15
10_15_21
9,18,27
11,11,12,12

"""
    )
    
    # ==================== New: Initial Channels ====================
    parser.add_argument(
        "--initial_channels",
        type=str,
        default="0,1,2,46,35,37,53,54",
        help="""Initial channel indices (overrides config file):
Format: comma-separated channel indices
Length: must equal base_channels
Example: --initial_channels 0,1,2,3,...,31

Scale 8 indices: [0,1,2,46,35,37,53,54]
Scale 4 indices: [0,1,2,4,5,6,7,8,9,10,44,46,35,37,53,54]
Scale 2 indices: [0,1,2,4,5,6,7,8,9,10,12,13,15,16,17,18,19,20,22,24,25,26,28,29,30,31,44,46,35,37,53,54]"""
    )
    
    # ==================== Scheduling Policy Selection (Retained for Quick Selection) ====================
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help="""Predefined scheduling policy (can be overridden by --selector and --step_sizes):
  uniform_10, uniform_5, progressive_5_10_15, etc.
Default: None"""
    )
    
    # ==================== Model Selection ====================
    parser.add_argument(
        "--model",
        type=str,
        default="Conv_default",
        choices=[
            "Conv_default", "Conv_deep", "Conv_wide",
            "Transformer_default", "Transformer_deep", "Transformer_large",
            "MLP_default", "MLP_deep"
        ],
        help="Model architecture selection (default: Conv_default)"
    )
    
    # ==================== Training Configuration Selection ====================
    parser.add_argument(
        "--train_config",
        type=str,
        default="default",
        choices=["default", "transformer", "fast", "no_early_stop"],
        help="Training hyperparameter configuration (default: default)"
    )
    
    # ==================== Hyperparameter Override Options ====================
    override_group = parser.add_argument_group("Hyperparameter Override")
    
    override_group.add_argument("--lr", type=float, default=None, help="Learning rate")
    override_group.add_argument("--batch_size", type=int, default=None, help="Batch size")
    override_group.add_argument("--epochs", type=int, default=None, help="Training epochs")
    override_group.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    override_group.add_argument("--seed", type=int, default=None, help="Random seed")
    
    # ==================== Runtime Options ====================
    runtime_group = parser.add_argument_group("Runtime Options")
    
    runtime_group.add_argument(
        "--device",
        type=str,
        default="cuda:0,1,2",
        help="Computing device (cpu, cuda:0, cuda:0,1, etc.)"
    )
    
    runtime_group.add_argument(
        "--run_root",
        type=str,
        default=None,
        help="Root directory for saving run results (default: runs)"
    )
    
    # ==================== Utility Options ====================
    util_group = parser.add_argument_group("Utility Options")
    
    util_group.add_argument(
        "--list_configs",
        action="store_true",
        help="List all available configuration options and exit"
    )
    
    util_group.add_argument(
        "--dry_run",
        action="store_true",
        help="Display configuration only without executing training"
    )
    
    return parser


def parse_device_config(device_str):
    """Parse device configuration string"""
    if device_str.lower() == 'cpu':
        return torch.device('cpu'), None, False
    
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, falling back to CPU")
        return torch.device('cpu'), None, False
    
    if ',' in device_str:
        device_ids = [int(x) for x in device_str.replace('cuda:', '').split(',')]
        num_gpus = torch.cuda.device_count()
        for gpu_id in device_ids:
            if gpu_id >= num_gpus:
                raise ValueError(f"GPU {gpu_id} not available. Total GPUs: {num_gpus}")
        device = torch.device(f'cuda:{device_ids[0]}')
        return device, device_ids, True
    else:
        if ':' in device_str:
            gpu_id = int(device_str.split(':')[1])
        else:
            gpu_id = 0
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus:
            raise ValueError(f"GPU {gpu_id} not available. Total GPUs: {num_gpus}")
        device = torch.device(f'cuda:{gpu_id}')
        return device, None, False


def print_device_info(device, device_ids, is_multi_gpu):
    """Print device information"""
    print("\n" + "=" * 80)
    print("Device Configuration")
    print("=" * 80)
    
    if device.type == 'cpu':
        print("Training on CPU")
    elif is_multi_gpu:
        print(f"Training on Multiple GPUs: {device_ids}")
        print(f"Primary device: {device}")
        print("-" * 80)
        for gpu_id in device_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            print(f"  GPU {gpu_id}: {props.name}")
            print(f"    Memory: {props.total_memory / (1024**3):.1f} GB")
        print("-" * 80)
        print(f"Total GPUs: {len(device_ids)}")
    else:
        gpu_id = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"Training on Single GPU: {gpu_id}")
        print(f"Device: {props.name}")
        print(f"Memory: {props.total_memory / (1024**3):.1f} GB")
    
    print("=" * 80)


def wrap_model_for_multi_gpu(model, device, device_ids):
    """Wrap model for multi-GPU training"""
    if device_ids is not None and len(device_ids) > 1:
        print(f"\n🔧 Wrapping model with DataParallel on GPUs: {device_ids}")
        model = DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    return model


def compute_nmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Normalized Mean Squared Error (NMSE)"""
    mse = np.mean((pred - true) ** 2)
    var_true = np.var(true)
    nmse = mse / (var_true + 1e-10)
    return nmse


def compute_pcc(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient (PCC)"""
    # Flatten to 1D
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    # Compute correlation coefficient
    correlation_matrix = np.corrcoef(pred_flat, true_flat)
    pcc = correlation_matrix[0, 1]
    
    return pcc


def compute_snr(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in dB"""
    signal_power = np.mean(true ** 2)
    noise_power = np.mean((pred - true) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr


def test_model(model, test_data, scheduler, selector, norm_stats, device, config):
    """
    Evaluate model on test set (one-time evaluation after training)
    
    Args:
        model: Trained model
        test_data: Test data (N, C_total, L) - original dimensions
        scheduler: Scheduler
        selector: Selector
        norm_stats: Normalization statistics
        device: Device
        config: Configuration dictionary
    
    Returns:
        test_results: Dictionary containing all test metrics
        reconstructed: Reconstructed data (N, C_total, L) - maintains original index positions
        ground_truth: Ground truth data (N, C_total, L)
    """
    print("\n" + "=" * 80)
    print("Final Testing on test set...")
    print("=" * 80)
    
    model.eval()
    
    # Extract configuration parameters
    base_channels = config['sr']['base_channels']
    target_channels = config['sr']['target_channels']
    step_sizes = config['schedule']['step_sizes']
    
    # 🔧 Get initial channel and target channel indices
    initial_channels = config['sr'].get('initial_channels', list(range(base_channels)))
    target_channel_indices = config['sr'].get('target_channels_indices', list(range(target_channels)))
    
    # Compute indices of channels to be generated
    initial_set = set(initial_channels)
    generated_channel_indices = [ch for ch in target_channel_indices if ch not in initial_set]
    
    print(f"\n📌 Test Configuration:")
    print(f"  Initial channels: {len(initial_channels)} channels at positions {initial_channels[:5]}...{initial_channels[-5:]}")
    print(f"  Target channels: {len(target_channel_indices)} channels")
    print(f"  Generated channels: {len(generated_channel_indices)} channels")
    print(f"  Step sizes: {step_sizes}")
    
    # Normalize test data
    from utils.norm import normalize, denormalize
    test_norm = normalize(test_data.astype(np.float32), norm_stats)
    test_norm_t = torch.from_numpy(test_norm).to(device)
    
    N, C_total, L = test_data.shape
    
    # 🔧 Initialize reconstruction results (maintain original dimensions and index positions)
    recon_norm = torch.zeros(N, C_total, L, device=device, dtype=test_norm_t.dtype)
    
    # Copy initial channel data at original positions
    for orig_idx in initial_channels:
        recon_norm[:, orig_idx, :] = test_norm_t[:, orig_idx, :]
    
    print(f"\n✓ Initialized reconstruction at original positions")
    
    # 🔧 Autoregressive generation
    batch_size = config['train']['batch_size']
    max_in_channels = target_channels - min(step_sizes)
    
    print(f"\nStarting autoregressive generation ({len(step_sizes)} steps)...")
    
    with torch.no_grad():
        current_channels = list(initial_channels)
        gen_idx = 0
        
        for s in range(len(step_sizes)):
            step_size = step_sizes[s]
            step_gen_indices = generated_channel_indices[gen_idx:gen_idx + step_size]
            
            print(f"Step {s+1}/{len(step_sizes)}: Generating {step_size} channels at positions {step_gen_indices}")
            
            # Process in batches
            for start_idx in range(0, N, batch_size):
                end_idx = min(N, start_idx + batch_size)
                batch_N = end_idx - start_idx
                
                # Construct input (collect from known channel positions)
                inp = torch.zeros(batch_N, max_in_channels, L, device=device, dtype=test_norm_t.dtype)
                
                for i, ch_idx in enumerate(current_channels):
                    inp[:, i, :] = recon_norm[start_idx:end_idx, ch_idx, :]
                
                # Model prediction
                out = model(inp)
                
                # Place predictions at corresponding original positions
                for i, ch_idx in enumerate(step_gen_indices):
                    recon_norm[start_idx:end_idx, ch_idx, :] = out[:, i, :]
            
            # Update known channel list
            current_channels.extend(step_gen_indices)
            gen_idx += step_size
    
    print("✓ Autoregressive generation completed!")
    
    # 🔧 Denormalize
    recon = denormalize(recon_norm.cpu().numpy(), norm_stats)
    
    # Replace initial channels at original positions with ground truth
    for orig_idx in initial_channels:
        recon[:, orig_idx, :] = test_data[:, orig_idx, :]
    
    print("✓ Reconstruction denormalized and initial channels replaced")
    
    # ==================== Compute Metrics ====================
    print("\n" + "=" * 80)
    print("Computing metrics...")
    print("=" * 80)
    
    # 🔧 1. Metrics for all target channels
    recon_all = recon[:, target_channel_indices, :]
    test_all = test_data[:, target_channel_indices, :]
    
    # Old metrics
    mse_all = np.mean((recon_all - test_all) ** 2)
    rmse_all = np.sqrt(mse_all)
    mae_all = np.mean(np.abs(recon_all - test_all))
    
    # New metrics
    nmse_all = compute_nmse(recon_all, test_all)
    pcc_all = compute_pcc(recon_all, test_all)
    snr_all = compute_snr(recon_all, test_all)
    
    # 🔧 2. Metrics for generated channels only
    recon_gen = recon[:, generated_channel_indices, :]
    test_gen = test_data[:, generated_channel_indices, :]
    
    # Old metrics
    mse_gen = np.mean((recon_gen - test_gen) ** 2)
    rmse_gen = np.sqrt(mse_gen)
    mae_gen = np.mean(np.abs(recon_gen - test_gen))
    
    # New metrics
    nmse_gen = compute_nmse(recon_gen, test_gen)
    pcc_gen = compute_pcc(recon_gen, test_gen)
    snr_gen = compute_snr(recon_gen, test_gen)
    
    # Compute metrics per channel (target channels)
    mse_per_channel = np.mean((recon_all - test_all) ** 2, axis=(0, 2))
    rmse_per_channel = np.sqrt(mse_per_channel)
    
    # Build results dictionary
    test_results = {
        # All target channels (after replacement)
        'all_channels': {
            'mse': float(mse_all),
            'rmse': float(rmse_all),
            'mae': float(mae_all),
            'nmse': float(nmse_all),
            'pcc': float(pcc_all),
            'snr_db': float(snr_all),
            'num_channels': len(target_channel_indices)
        },
        # Generated channels only
        'generated_channels': {
            'mse': float(mse_gen),
            'rmse': float(rmse_gen),
            'mae': float(mae_gen),
            'nmse': float(nmse_gen),
            'pcc': float(pcc_gen),
            'snr_db': float(snr_gen),
            'num_channels': len(generated_channel_indices)
        },
        # Other information
        'per_channel': {
            'mse': mse_per_channel.tolist(),
            'rmse': rmse_per_channel.tolist()
        },
        'metadata': {
            'num_samples': int(N),
            'initial_channels': initial_channels,
            'target_channel_indices': target_channel_indices,
            'generated_channel_indices': generated_channel_indices,
            'reconstructed_shape': list(recon.shape),
            'ground_truth_shape': list(test_data.shape)
        }
    }
    
    # ==================== Print Results ====================
    print(f"\n{'=' * 80}")
    print("Final Test Results:")
    print(f"{'=' * 80}")
    
    print(f"\n📊 1. All Target Channels ({len(target_channel_indices)} channels, after replacement):")
    print(f"  {'Metric':<10} {'Value':<15}")
    print(f"  {'-'*25}")
    print(f"  {'MSE':<10} {mse_all:<15.6e}")
    print(f"  {'RMSE':<10} {rmse_all:<15.6f}")
    print(f"  {'MAE':<10} {mae_all:<15.6f}")
    print(f"  {'NMSE':<10} {nmse_all:<15.6f}")
    print(f"  {'PCC':<10} {pcc_all:<15.6f}")
    print(f"  {'SNR (dB)':<10} {snr_all:<15.2f}")
    
    print(f"\n📊 2. Generated Channels Only ({len(generated_channel_indices)} channels):")
    print(f"  {'Metric':<10} {'Value':<15}")
    print(f"  {'-'*25}")
    print(f"  {'MSE':<10} {mse_gen:<15.6e}")
    print(f"  {'RMSE':<10} {rmse_gen:<15.6f}")
    print(f"  {'MAE':<10} {mae_gen:<15.6f}")
    print(f"  {'NMSE':<10} {nmse_gen:<15.6f}")
    print(f"  {'PCC':<10} {pcc_gen:<15.6f}")
    print(f"  {'SNR (dB)':<10} {snr_gen:<15.2f}")
    
    print(f"\n  Total samples: {N}")
    print(f"{'=' * 80}\n")
    
    return test_results, recon, test_data


def load_checkpoint_smart(model, checkpoint_path, device, verbose=True):
    """
    Smartly load model checkpoint, automatically handles 'module.' prefix issue with DataParallel
    
    Supports all following scenarios:
    1. Checkpoint has 'module.', current model has DataParallel
    2. Checkpoint has 'module.', current model doesn't have DataParallel
    3. Checkpoint doesn't have 'module.', current model has DataParallel
    4. Checkpoint doesn't have 'module.', current model doesn't have DataParallel
    
    Args:
        model: Current model (may be wrapped by DataParallel)
        checkpoint_path: Checkpoint file path
        device: Device
        verbose: Whether to print detailed information
    
    Returns:
        checkpoint: Complete checkpoint dictionary
    """
    if verbose:
        print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Analyze checkpoint
    checkpoint_keys = list(state_dict.keys())
    checkpoint_has_module = checkpoint_keys[0].startswith('module.') if checkpoint_keys else False
    
    # Analyze current model
    model_is_parallel = isinstance(model, DataParallel)
    
    # Get target model (may be inside DataParallel)
    if model_is_parallel:
        target_model = model.module
    else:
        target_model = model
    
    target_keys = list(target_model.state_dict().keys())
    
    if verbose:
        print(f"  Checkpoint info:")
        print(f"    - Has 'module.' prefix: {checkpoint_has_module}")
        print(f"    - Sample keys: {checkpoint_keys[:2]}")
        print(f"  Current model info:")
        print(f"    - Is DataParallel: {model_is_parallel}")
        print(f"    - Sample keys: {target_keys[:2]}")
    
    # Conversion logic: Ensure checkpoint and model key formats match
    if checkpoint_has_module:
        # Checkpoint has 'module.', remove it
        new_state_dict = {
            k.replace('module.', '', 1): v 
            for k, v in state_dict.items()
        }
        if verbose:
            print(f"  🔧 Converting: Removing 'module.' prefix from checkpoint")
    else:
        # Checkpoint doesn't have 'module.', keep as is
        new_state_dict = state_dict
    
    # Load to target model (regardless of DataParallel wrapping)
    try:
        target_model.load_state_dict(new_state_dict, strict=True)
        if verbose:
            print(f"  ✓ Successfully loaded checkpoint!")
        return checkpoint
    
    except RuntimeError as e:
        print(f"\n❌ Error: Failed to load checkpoint!")
        print(f"  Error message: {str(e)[:200]}...")
        print(f"\n  Debug info:")
        print(f"    Checkpoint 'module.' prefix: {checkpoint_has_module}")
        print(f"    Model is DataParallel: {model_is_parallel}")
        print(f"    Target model keys (first 3): {target_keys[:3]}")
        print(f"    State dict keys (first 3): {list(new_state_dict.keys())[:3]}")
        raise RuntimeError(f"Failed to load checkpoint: {str(e)[:100]}")


def main():
    # Parse command-line arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # ==================== List Configuration Options ====================
    if args.list_configs:
        list_seed_configs(args.master_config)
        return
    
    # ==================== Build Configuration ====================
    print("=" * 80)
    print("Building experiment configuration...")
    print("=" * 80)
    
    # Validate parameter combinations
    if args.schedule is None and (args.selector is None or args.step_sizes is None):
        parser.error("Must specify either --schedule OR (--selector AND --step_sizes)")
    
    # If using command-line arguments, build temporary schedule configuration
    if args.selector is not None or args.step_sizes is not None:
        schedule_name = 'custom'
        
        # Parse step_sizes
        if args.step_sizes:
            step_sizes = [int(x.strip()) for x in args.step_sizes.split(',')]
        else:
            parser.error("--step_sizes is required when using --selector")
        
        # Parse selector
        selector_name = args.selector if args.selector else 'sequential'
        
        # Build selector_params
        selector_params = {}
        if selector_name == 'geometric':
            selector_params['electrode_positions'] = args.electrode_positions
            selector_params['mode'] = args.selector_mode if args.selector_mode else 'uniform'
        elif selector_name == 'matrix':
            selector_params['mode'] = args.selector_mode if args.selector_mode else 'greedy'
        elif selector_name == 'random':
            selector_params['seed'] = args.selector_seed
        elif selector_name == 'custom':
            if args.custom_order is None:
                parser.error("--custom_order is required when using --selector custom")
            selector_params['channel_order'] = [int(x.strip()) for x in args.custom_order.split(',')]
        
        custom_schedule = {
            'step_sizes': step_sizes,
            'selector': selector_name,
            'selector_params': selector_params
        }
        
        print(f"\n✓ Using custom schedule configuration:")
        print(f"  Selector: {selector_name}")
        print(f"  Step sizes: {step_sizes}")
        if selector_params:
            print(f"  Selector params: {selector_params}")
    else:
        schedule_name = args.schedule
        custom_schedule = None
    
    # Build base configuration
    config = build_seed_config(
        master_config_path=args.master_config,
        data_source=args.data_source,
        sr_ratio=args.sr_ratio,
        schedule=schedule_name if custom_schedule is None else 'uniform_10',
        model=args.model,
        train_config=args.train_config
    )
    
    # If using custom configuration, override schedule section
    if custom_schedule:
        config['schedule'] = custom_schedule
        # Add identifier for folder naming
        config['schedule_name'] = f"{custom_schedule['selector']}_{'_'.join(map(str, custom_schedule['step_sizes']))}"
    
    # Handle initial_channels
    if args.initial_channels:
        initial_channels = [int(x.strip()) for x in args.initial_channels.split(',')]
        expected_len = config['sr']['base_channels']
        if len(initial_channels) != expected_len:
            parser.error(f"--initial_channels length must be {expected_len}, got {len(initial_channels)}")
        target_channels = config['sr']['target_channels']
        if any(ch < 0 or ch >= target_channels for ch in initial_channels):
            parser.error(f"--initial_channels must be in range [0, {target_channels})")
        if len(set(initial_channels)) != len(initial_channels):
            parser.error("--initial_channels must not have duplicates")
        config['sr']['initial_channels'] = initial_channels
        print(f"\n✓ Using custom initial channels: {initial_channels[:5]}...{initial_channels[-5:]}")
    
    print(f"\nExperiment: SEED/{args.data_source}/{args.sr_ratio}/{args.model}/{config['schedule']['selector']}")
    print(f"Train config: {args.train_config}")
    
    # ==================== Apply Command-Line Overrides ====================
    overrides = []
    if args.lr is not None:
        config['train']['lr'] = args.lr
        overrides.append(f"lr={args.lr}")
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
        overrides.append(f"batch_size={args.batch_size}")
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
        overrides.append(f"epochs={args.epochs}")
    if args.weight_decay is not None:
        config['train']['weight_decay'] = args.weight_decay
        overrides.append(f"weight_decay={args.weight_decay}")
    if args.seed is not None:
        config['train']['seed'] = args.seed
        overrides.append(f"seed={args.seed}")
    if args.run_root is not None:
        config['runtime']['run_root'] = args.run_root
        overrides.append(f"run_root={args.run_root}")
    
    if overrides:
        print(f"\nParameter overrides: {', '.join(overrides)}")
    
    # ==================== Dry Run ====================
    if args.dry_run:
        print("\n" + "=" * 80)
        print("Configuration (dry run, not executing training):")
        print("=" * 80)
        import json
        print(json.dumps(config, indent=2))
        return
    
    # ==================== Set Random Seed ====================
    set_seed(config['train']['seed'], config['train']['deterministic'])
    print(f"\nRandom seed set to: {config['train']['seed']}")
    
    # ==================== Parse Device Configuration ====================
    device, device_ids, is_multi_gpu = parse_device_config(args.device)
    config['runtime']['device'] = str(device)
    config['runtime']['device_ids'] = device_ids
    config['runtime']['is_multi_gpu'] = is_multi_gpu
    config['runtime']['num_gpus'] = len(device_ids) if device_ids else (1 if device.type == 'cuda' else 0)
    print_device_info(device, device_ids, is_multi_gpu)
    
    # ==================== Create Run Directory ====================
    run_dir = create_run_dir(config)
    save_json(config, os.path.join(run_dir, "config.json"))
    print(f"\nRun directory: {run_dir}")
    
    # ==================== Load Data ====================
    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)
    
    train_data, test_data, meta = load_data(config)
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # ==================== Compute Normalization Statistics ====================
    norm_stats = compute_norm_stats(train_data, mode=config['data']['normalization'])
    save_json(norm_stats, os.path.join(run_dir, "norm_stats.json"))
    print(f"Normalization mode: {config['data']['normalization']}")
    
    # ==================== Build Scheduler and Selector ====================
    print("\n" + "=" * 80)
    print("Building scheduler and selector...")
    print("=" * 80)
    
    selector = get_selector(
        config['schedule']['selector'],
        config['schedule']['selector_params']
    )
    
    initial_channels = config['sr'].get('initial_channels', None)
    
    scheduler = StepScheduler(
        base_channels=config['sr']['base_channels'],
        target_channels=config['sr']['target_channels'],
        step_sizes=config['schedule']['step_sizes'],
        selector=selector,
        initial_channels=initial_channels
    )
    
    print(f"\nChannel generation schedule:")
    print(f"  Base channels: {config['sr']['base_channels']}")
    print(f"  Target channels: {config['sr']['target_channels']}")
    if initial_channels:
        print(f"  Initial channels: custom ({len(initial_channels)} channels)")
        print(f"    First 5: {initial_channels[:5]}")
        print(f"    Last 5: {initial_channels[-5:]}")
    else:
        print(f"  Initial channels: default [0, 1, ..., {config['sr']['base_channels']-1}]")
    print(f"  Step sizes: {config['schedule']['step_sizes']}")
    print(f"  Selector: {config['schedule']['selector']}")
    if config['schedule']['selector_params']:
        print(f"  Selector params: {config['schedule']['selector_params']}")
    print("=" * 80 + "\n")
    
    # ==================== Build Datasets (No Validation Split) ====================
    print("\n" + "=" * 80)
    print("Building datasets...")
    print("=" * 80)
    print("⚠️  Note: Using test set as validation set (no split from train data)")
    
    # Training set: Use all training data
    train_dataset = EEGChannelARDataset(train_data, scheduler, selector, norm_stats)
    
    # Validation set: Use test data as validation set
    val_dataset = EEGChannelARDataset(test_data, scheduler, selector, norm_stats)
    
    batch_size = config['train']['batch_size']
    num_workers = 4 if device.type == 'cuda' else 0
    pin_memory = (device.type == 'cuda')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Train samples: {len(train_dataset)} (100% of train data)")
    print(f"Validation samples: {len(val_dataset)} (test data used as validation)")
    print(f"Batch size: {batch_size}")
    if is_multi_gpu:
        print(f"Effective batch size per GPU: {batch_size // len(device_ids)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # ==================== Build Model ====================
    print("\n" + "=" * 80)
    print("Building model...")
    print("=" * 80)
    
    max_in_channels = config['sr']['target_channels'] - min(config['schedule']['step_sizes'])
    out_channels = max(config['schedule']['step_sizes'])
    model = get_model(config, in_channels=max_in_channels, out_channels=out_channels)
    
    print(f"Model type: {config['model']['name']}")
    print(f"Input channels: {max_in_channels}")
    print(f"Output channels: {out_channels}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== Wrap Model (Multi-GPU Support) ====================
    model = wrap_model_for_multi_gpu(model, device, device_ids)
    
    # ==================== Training ====================
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    print(f"Epochs: {config['train']['epochs']}")
    print(f"Learning rate: {config['train']['lr']}")
    print(f"Weight decay: {config['train']['weight_decay']}")
    print(f"Early stopping patience: {config['train']['early_stopping']['patience']}")
    print(f"⚠️  Validation = Test set (for early stopping and model selection)")
    print("=" * 80 + "\n")
    
    trainer = Trainer(model, config, device, run_dir)
    trainer.fit(train_loader, val_loader, epochs=config['train']['epochs'])
    
    # ==================== Load Best Model and Final Testing After Training ====================
    print("\n" + "=" * 80)
    print("Loading best model for final testing...")
    print("=" * 80)
    
    # 🔧 Use smart loading function
    best_model_path = os.path.join(run_dir, "checkpoints", "best_model.pth")
    checkpoint = load_checkpoint_smart(model, best_model_path, device, verbose=True)
    
    print(f"✓ Loaded best model from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.6f})")
    
    # 🔧 Final evaluation on test set (using fixed test_model)
    test_results, reconstructed, ground_truth = test_model(
        model, test_data, scheduler, selector, norm_stats, device, config
    )
    
    # Save test results
    save_json(test_results, os.path.join(run_dir, "test_results.json"))
    
    # Save reconstructed data (full dimensions)
    np.save(os.path.join(run_dir, "test_reconstructed.npy"), reconstructed)
    np.save(os.path.join(run_dir, "test_ground_truth.npy"), ground_truth)
    
    print(f"\n✓ Test results saved to: {run_dir}/test_results.json")
    print(f"✓ Reconstructed data saved to: {run_dir}/test_reconstructed.npy (shape: {reconstructed.shape})")
    print(f"✓ Ground truth saved to: {run_dir}/test_ground_truth.npy")
    
    # ==================== Completion ====================
    print("\n" + "=" * 80)
    print("Training and Testing completed!")
    print("=" * 80)
    print(f"Best model: {run_dir}/checkpoints/best_model.pth")
    print(f"Training logs: {run_dir}/logs/")
    print(f"Test results: {run_dir}/test_results.json")
    print(f"Configuration: {run_dir}/config.json")
    
    if is_multi_gpu:
        print(f"\n💡 Note: Model was trained with DataParallel on {len(device_ids)} GPUs")
    
    print(f"\n⚠️  Important: Test set was used as validation set during training")
    print(f"⚠️  Reconstructed data maintains original channel dimensions and positions")
    print("=" * 80)


if __name__ == "__main__":
    main()