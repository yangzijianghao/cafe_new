import os
import json
from datetime import datetime


def create_run_dir(config: dict) -> str:
    """
    Create hierarchical run directory based on configuration
    Format: runs/{dataset_name}/{data_name}/{split}/{sr_ratio}/{model}/{selector}/{schedule}/{timestamp}
    """
    root = config['runtime']['run_root']
    dataset_name = config['data'].get('dataset_name', 'unknown')  # New: dataset name
    data_name = config['data']['name']
    split = config['split']['type']
    base = config['sr']['base_channels']
    target = config['sr']['target_channels']
    sr_ratio = f"{base}to{target}"
    model = config['model']['name']
    selector = config['schedule']['selector']
    steps = config['schedule']['step_sizes']
    schedule_str = "_".join(map(str, steps))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = os.path.join(
        root, dataset_name, data_name, split, sr_ratio, model, selector, schedule_str, timestamp
    )
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "test"), exist_ok=True)
    
    return run_dir


def save_json(data: dict, path: str):
    """Atomically save JSON"""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_json(path: str) -> dict:
    """Load JSON"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)