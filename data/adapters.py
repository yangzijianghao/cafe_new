import os
import numpy as np


def load_data(config: dict) -> tuple:
    """
    Unified data loading interface.
    Returns: (train_data, test_data, meta)
    train/test: (N, C, L) ndarray
    meta: dict, optional channel positions / label information
    """
    data_root = config['data']['root']
    train_file = config['data']['train_file']
    test_file = config['data']['test_file']
    
    train_path = os.path.join(data_root, train_file)
    test_path = os.path.join(data_root, test_file)
    
    train_data = np.load(train_path).astype(np.float32)
    test_data = np.load(test_path).astype(np.float32)
    
    # meta can be extended to include channel positions, labels, etc.
    meta = {}
    
    return train_data, test_data, meta
