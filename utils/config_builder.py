import copy
from utils.io import load_json


def build_seed_config(
    master_config_path: str,
    data_source: str,
    sr_ratio: str,
    schedule: str,
    model: str,
    train_config: str = "default"
) -> dict:
    """
    Build complete experiment configuration from SEED master configuration file
    
    Args:
        master_config_path: Path to SEED master configuration file
        data_source: Data source, e.g., "random_split" or "cross_subject"
        sr_ratio: Super-resolution ratio, e.g., "half", "quarter", "eighth"
        schedule: Scheduling policy, e.g., "uniform_10", "progressive_5_10_15"
        model: Model configuration name, e.g., "Conv_default", "Transformer_default"
        train_config: Training configuration, e.g., "default", "transformer", "fast"
    
    Returns:
        Complete experiment configuration dictionary
    """
    master = load_json(master_config_path)
    
    # Validate configuration existence
    assert data_source in master['data_sources'], f"Data source '{data_source}' not found"
    assert sr_ratio in master['sr_ratios'], f"SR ratio '{sr_ratio}' not found"
    assert schedule in master['schedules'], f"Schedule '{schedule}' not found"
    assert model in master['models'], f"Model '{model}' not found"
    assert train_config in master['train_configs'], f"Train config '{train_config}' not found"
    
    # Build configuration
    config = {
        "data": {
            "dataset_name": master['dataset_info']['name'],
            "name": data_source,  # Add this field for path naming
            **master['data_sources'][data_source]
        },
        "split": {
            "type": data_source  # Retain for logical judgment
        },
        "sr": master['sr_ratios'][sr_ratio],
        "schedule": master['schedules'][schedule],
        "model": master['models'][model],
        "train": master['train_configs'][train_config],
        "runtime": master['runtime']
    }
    
    return config


def list_seed_configs(master_config_path: str):
    """List all available configuration options for SEED dataset"""
    master = load_json(master_config_path)
    print("=" * 60)
    print(f"SEED Dataset Configuration Options")
    print("=" * 60)
    print(f"\nDataset: {master['dataset_info']['name']}")
    print(f"Total Channels: {master['dataset_info']['total_channels']}")
    print(f"Sampling Rate: {master['dataset_info']['sampling_rate']} Hz")
    print(f"\nAvailable configurations:")
    print(f"  Data Sources: {list(master['data_sources'].keys())}")
    print(f"  SR Ratios: {list(master['sr_ratios'].keys())}")
    print(f"  Schedules: {list(master['schedules'].keys())}")
    print(f"  Models: {list(master['models'].keys())}")
    print(f"  Train Configs: {list(master['train_configs'].keys())}")
    print("=" * 60)