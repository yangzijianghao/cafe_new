import numpy as np
import json
import os


class SequentialSelector:
    """Select channels sequentially by index order."""
    def __init__(self, params: dict = None):
        pass

    def select(self, known_channels: np.ndarray, candidate_channels: np.ndarray, n: int) -> np.ndarray:
        """
        Select the first n candidate channels in order.

        Args:
            known_channels: (K,) Indices of known channels
            candidate_channels: (M,) Indices of candidate channels
            n: Number of channels to select

        Returns:
            (n,) Selected channel indices (order preserved)
        """
        assert len(candidate_channels) >= n, f"Not enough candidate channels: {len(candidate_channels)} < {n}"
        return candidate_channels[:n]


class MatrixDistanceSelector:
    """
    Index-distance-based selector on a matrix layout.

    Principle: Greedily select the channel that is closest to existing channels.
    Distance definition: min |idx_new - idx_known| over all known channels
    """
    def __init__(self, params: dict = None):
        if params is None:
            params = {}
        self.mode = params.get('mode', 'greedy')  # 'greedy' or 'balanced'

    def select(self, known_channels: np.ndarray, candidate_channels: np.ndarray, n: int) -> np.ndarray:
        """
        Greedy selection based on index distance.

        Strategy: At each step, select the candidate channel with the minimum distance to known channels.
        """
        assert len(candidate_channels) >= n, f"Not enough candidate channels: {len(candidate_channels)} < {n}"

        selected = []
        remaining = candidate_channels.copy()
        current_known = known_channels.copy()

        for _ in range(n):
            # Compute the minimum distance from each candidate to known channels
            min_distances = []
            for cand in remaining:
                distances = np.abs(cand - current_known)
                min_distances.append(distances.min())

            min_distances = np.array(min_distances)

            if self.mode == 'greedy':
                # Select the closest channel
                best_idx = np.argmin(min_distances)
            elif self.mode == 'balanced':
                # Select a medium-distance channel to avoid clustering
                median_dist = np.median(min_distances)
                best_idx = np.argmin(np.abs(min_distances - median_dist))
            else:
                best_idx = np.argmin(min_distances)

            selected_channel = remaining[best_idx]
            selected.append(selected_channel)

            # Update known and remaining sets
            current_known = np.append(current_known, selected_channel)
            remaining = np.delete(remaining, best_idx)

        # Preserve original index order
        selected = np.array(selected, dtype=np.int64)
        selected.sort()
        return selected


class GeometricDistanceSelector:
    """
    Geometry-based selector using electrode positions.

    Requires electrode coordinates (2D or 3D).
    """
    def __init__(self, params: dict):
        positions_path = params.get('electrode_positions', None)
        if positions_path is None:
            raise ValueError("GeometricDistanceSelector requires 'electrode_positions' parameter")

        if not os.path.isabs(positions_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            positions_path = os.path.join(project_root, positions_path)

        if not os.path.exists(positions_path):
            raise FileNotFoundError(f"Electrode positions file not found: {positions_path}")

        with open(positions_path, 'r') as f:
            data = json.load(f)

        # Expected format:{"channels": [...], "positions": [[x,y,z], ...]}
        self.channel_names = data.get('channels', None)
        self.positions = np.array(data['positions'], dtype=np.float32)  # (C, 3) or (C, 2)

        self.mode = params.get('mode', 'greedy')  # 'greedy' or 'uniform'

        print(f"Loaded {len(self.positions)} electrode positions from {positions_path}")

    def select(self, known_channels: np.ndarray, candidate_channels: np.ndarray, n: int) -> np.ndarray:
        """
        Greedy selection based on geometric distance.

        Strategy: At each step, select the candidate channel with the minimum Euclidean distance to known channels.
        """
        assert len(candidate_channels) >= n, f"Not enough candidate channels: {len(candidate_channels)} < {n}"

        selected = []
        remaining = candidate_channels.copy()
        current_known = known_channels.copy()

        for _ in range(n):
            # Compute minimum geometric distance to known channels
            min_distances = []
            for cand in remaining:
                cand_pos = self.positions[cand] # (3,) or (2,)
                known_pos = self.positions[current_known] # (K, 3) or (K, 2)

                # Euclidean distance
                distances = np.linalg.norm(known_pos - cand_pos, axis=1)
                min_dist = distances.min()
                min_distances.append(min_dist)

            min_distances = np.array(min_distances)

            if self.mode == 'greedy':
                # Select the closest channel (fill nearest gap)
                best_idx = np.argmin(min_distances)
            elif self.mode == 'uniform':
                # Select the farthest channel (promote uniform coverage)
                best_idx = np.argmax(min_distances)
            else:
                best_idx = np.argmin(min_distances)

            selected_channel = remaining[best_idx]
            selected.append(selected_channel)

            # Update known and remaining sets
            current_known = np.append(current_known, selected_channel)
            remaining = np.delete(remaining, best_idx)

        selected = np.array(selected, dtype=np.int64)
        selected.sort()  # Preserve index order
        return selected


class CustomOrderSelector:
    """
    Custom channel order selector.

    Allows users to define a complete channel generation order.
    """
    def __init__(self, params: dict):
        # params['channel_order']: full channel index sequence
        self.channel_order = np.array(params['channel_order'], dtype=np.int64)
        self.current_idx = 0

    def select(self, known_channels: np.ndarray, candidate_channels: np.ndarray, n: int) -> np.ndarray:
        """Select channels following a predefined order."""
        selected = []
        for _ in range(n):
            while self.current_idx < len(self.channel_order):
                ch = self.channel_order[self.current_idx]
                self.current_idx += 1
                if ch in candidate_channels:
                    selected.append(ch)
                    break

        assert len(selected) == n, f"Custom order exhausted: {len(selected)} < {n}"
        return np.array(selected, dtype=np.int64)


class RandomSelector:
    """Random channel selector (for baseline comparison)."""
    def __init__(self, params: dict = None):
        if params is None:
            params = {}
        self.seed = params.get('seed', None)
        if self.seed is not None:
            np.random.seed(self.seed)

    def select(self, known_channels: np.ndarray, candidate_channels: np.ndarray, n: int) -> np.ndarray:
        """Randomly select n candidate channels."""
        assert len(candidate_channels) >= n
        selected = np.random.choice(candidate_channels, size=n, replace=False)
        selected.sort()
        return selected


# ...existing code...

def get_selector(name: str, params: dict):
    """Selector factory function."""
    # Name mapping (CLI shorthand -> actual selector name)
    name_mapping = {
        'sequential': 'sequential',
        'random': 'random',
        'geometric': 'geometric_distance',
        'matrix': 'matrix_distance',
        'custom': 'custom_order',
        'geometric_distance': 'geometric_distance',
        'matrix_distance': 'matrix_distance',
        'custom_order': 'custom_order',
    }

    actual_name = name_mapping.get(name, name)

    if actual_name == "sequential":
        return SequentialSelector(params)
    elif actual_name == "matrix_distance":
        return MatrixDistanceSelector(params)
    elif actual_name == "geometric_distance":
        return GeometricDistanceSelector(params)
    elif actual_name == "custom_order":
        return CustomOrderSelector(params)
    elif actual_name == "random":
        return RandomSelector(params)
    else:
        available = list(name_mapping.keys())
        raise ValueError(
            f"Unknown selector: {name} (mapped to: {actual_name})\n"
            f"Available selectors: {available}"
        )
