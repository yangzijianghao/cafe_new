import numpy as np


class StepScheduler:
    """
    Channel generation scheduler: computes the channel indices to be generated at each step based on step sizes and a selector.
    """
    def __init__(self, base_channels: int, target_channels: int, step_sizes: list, 
                 selector=None, initial_channels=None):
        """
        Args:
            base_channels: Number of initial channels (e.g., 32)
            target_channels: Number of target channels (e.g., 62)
            step_sizes: List specifying how many channels to generate at each step
            selector: Channel selector
            initial_channels: Explicit indices of initial channels (optional)
                - None: use default [0, 1, 2, ..., base_channels-1]
                - list/array: custom initial channel indices, length must equal base_channels
        """
        self.base_channels = base_channels
        self.target_channels = target_channels
        self.step_sizes = step_sizes
        self.num_steps = len(step_sizes)
        self.selector = selector
        
        # New: handle initial channels
        if initial_channels is None:
            self.initial_channels = np.arange(base_channels, dtype=np.int64)
        else:
            self.initial_channels = np.array(initial_channels, dtype=np.int64)
            assert len(self.initial_channels) == base_channels, \
                f"initial_channels length {len(self.initial_channels)} != base_channels {base_channels}"
            assert len(np.unique(self.initial_channels)) == base_channels, \
                "Duplicate indices found in initial_channels"
            assert self.initial_channels.min() >= 0 and self.initial_channels.max() < target_channels, \
                f"initial_channels indices out of range [0, {target_channels})"
        
        print(f"Initial channels ({len(self.initial_channels)}): {self.initial_channels[:10].tolist()}" + 
              (f"...{self.initial_channels[-3:].tolist()}" if len(self.initial_channels) > 10 else ""))
        
        # Validation
        total = sum(step_sizes)
        expected = target_channels - base_channels
        assert total == expected, f"sum(step_sizes)={total} != expected={expected}"
        
        # Pre-compute output channel indices for each step
        self._compute_step_indices()
    
    def _compute_step_indices(self):
        """Compute the exact channel indices to be generated at each step."""
        self.step_indices = []
        known_channels = self.initial_channels.copy()  # use specified initial channels
        all_channels = np.arange(self.target_channels, dtype=np.int64)
        
        for step in range(self.num_steps):
            # Candidate channels = all channels - known channels
            candidate_channels = np.setdiff1d(all_channels, known_channels)
            
            # Use selector if provided
            if self.selector is not None:
                selected = self.selector.select(
                    known_channels=known_channels,
                    candidate_channels=candidate_channels,
                    n=self.step_sizes[step]
                )
            else:
                # Default: select in index order
                selected = candidate_channels[:self.step_sizes[step]]
            
            self.step_indices.append(selected)
            
            # Update known channels
            known_channels = np.concatenate([known_channels, selected])
        
        print(f"Step indices computed:")
        for i, idx in enumerate(self.step_indices):
            if len(idx) <= 10:
                print(f"  Step {i}: {len(idx)} channels, {idx.tolist()}")
            else:
                print(f"  Step {i}: {len(idx)} channels, {idx[:5].tolist()}...{idx[-5:].tolist()}")
    
    def get_step_out_indices(self, step: int) -> np.ndarray:
        """Return the channel indices to be generated at the given step."""
        return self.step_indices[step]
    
    def get_cin(self, step: int) -> int:
        """Return the number of known channels at the given step."""
        if step == 0:
            return self.base_channels
        else:
            return self.base_channels + sum(self.step_sizes[:step])
    
    def get_known_channels(self, step: int) -> np.ndarray:
        """Return the indices of known channels at the given step."""
        if step == 0:
            return self.initial_channels.copy()  # return specified initial channels
        else:
            known = [self.initial_channels]  # start from specified initial channels
            for s in range(step):
                known.append(self.step_indices[s])
            return np.concatenate(known)