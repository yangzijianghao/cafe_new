import numpy as np


def compute_norm_stats(data: np.ndarray, mode: str = 'global') -> dict:
    """
    Compute normalization statistics
    
    Args:
        data: Data with shape (N, C, L)
        mode: 'global' or 'per_channel'
    
    Returns:
        Dictionary containing normalization parameters
    """
    if mode == 'global':
        mean = float(data.mean())
        std = float(data.std())
        return {
            'mode': 'global',
            'mean': mean,
            'std': std
        }
    elif mode == 'per_channel':
        # Compute per channel
        channel_means = data.mean(axis=(0, 2)).tolist()  # (C,)
        channel_stds = data.std(axis=(0, 2)).tolist()    # (C,)
        return {
            'mode': 'per_channel',
            'channel_means': channel_means,
            'channel_stds': channel_stds
        }
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def normalize(data: np.ndarray, norm_stats: dict) -> np.ndarray:
    """
    Normalize data using statistics (deprecated, now dynamically normalized in Dataset.__getitem__)
    """
    # Keep this function for test data normalization in evaluator
    if norm_stats['mode'] == 'global':
        return (data - norm_stats['mean']) / (norm_stats['std'] )
    elif norm_stats['mode'] == 'per_channel':
        mean = np.array(norm_stats['channel_means']).reshape(1, -1, 1)
        std = np.array(norm_stats['channel_stds']).reshape(1, -1, 1)
        return (data - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown mode: {norm_stats['mode']}")


def denormalize(data: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Denormalize data"""
    if norm_stats['mode'] == 'global':
        return data * (norm_stats['std']) + norm_stats['mean']
    elif norm_stats['mode'] == 'per_channel':
        mean = np.array(norm_stats['channel_means']).reshape(1, -1, 1)
        std = np.array(norm_stats['channel_stds']).reshape(1, -1, 1)
        return data * (std) + mean
    else:
        raise ValueError(f"Unknown mode: {norm_stats['mode']}")