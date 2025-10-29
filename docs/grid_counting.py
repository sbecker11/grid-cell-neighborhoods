"""
Efficient 2D array positive value counting with automatic hardware acceleration detection.

This module provides functions that automatically detect and use the fastest
available hardware (GPU via JAX/PyTorch, or optimized CPU via NumPy) to
efficiently count positive values (> 0) in 2D arrays.

This module is optimized for the use case: counting positive int elements
in a 2D array with H rows by W cols, where H and W are > 0 and element values are >= 0.

QUICK START:
    >>> from grid_counting import SparseGrid
    >>> grid = SparseGrid(height=100, width=100)
    >>> grid.set_neighborhoods_to_value(locations=[(10, 10), (20, 20)], L=3, value=2)
    >>> count = grid.count_positive_valued_cells()
    >>> # Returns count of positive-valued cells within Manhattan distance L from all seeds
"""

import numpy as np

# Track which accelerator message has been printed to avoid duplicates
_accelerator_message_printed = False


class DenseGrid:
    """A dense 2D grid represented as a numpy array."""
    
    def __init__(self, height, width=None, initial_value=0):
        """
        Create a DenseGrid from dimensions or existing array.
        
        Args:
            height: Number of rows, or existing numpy array, or tuple (height, width)
            width: Number of columns (required if height is an integer)
            initial_value: Value to fill if creating new grid (default=0)
        """
        # Support legacy tuple format: DenseGrid((height, width))
        if width is None and isinstance(height, tuple):
            height, width = height
        
        # Support existing array: DenseGrid(array)
        if width is None:
            self.grid = np.asarray(height).copy()
            if self.grid.size == 0 or len(self.grid.shape) != 2 or self.grid.shape[0] <= 0 or self.grid.shape[1] <= 0:
                raise ValueError(f"Grid must be 2D with dimensions > 0, got shape {self.grid.shape}")
        else:
            # New grid from dimensions: DenseGrid(height, width)
            if height <= 0 or width <= 0:
                raise ValueError(f"Grid dimensions must be > 0, got {height}x{width}")
            self.grid = np.full((height, width), initial_value, dtype=np.int32)
    
    def set_locations_to_value(self, locations, value=2):
        """Set specified locations to a value."""
        import numpy as np
        
        if not locations:
            return self
        
        # Convert locations to numpy array for vectorized operations
        locations = np.asarray(locations)
        
        # Handle different input formats
        if locations.ndim == 1:
            # Single location: reshape to 2D
            locations = locations.reshape(1, -1)
        
        # Validate shape
        if locations.shape[1] != 2:
            raise ValueError("Locations must be a list of (row, col) tuples")
        
        # Filter out-of-bounds locations
        rows, cols = self.grid.shape
        valid_mask = ((locations[:, 0] >= 0) & (locations[:, 0] < rows) &
                      (locations[:, 1] >= 0) & (locations[:, 1] < cols))
        valid_locations = locations[valid_mask]
        
        if len(valid_locations) == 0:
            return self
        
        # Use advanced indexing to set values efficiently
        row_indices = valid_locations[:, 0]
        col_indices = valid_locations[:, 1]
        
        result = np.array(self.grid, copy=True)
        # If value is 0, set regardless of current value (clear/reset)
        # Otherwise, only set if cell is currently unset (0)
        if value == 0:
            result[row_indices, col_indices] = value
        else:
            # Only set unset cells (value == 0)
            mask = result[row_indices, col_indices] == 0
            if np.any(mask):
                result[row_indices[mask], col_indices[mask]] = value
        self.grid = result
        return self
    
    def set_neighborhoods_to_value(self, locations, L=3, value=2):
        """Set neighborhoods around seeds using BFS."""
        from collections import deque
        
        result = np.array(self.grid, copy=True)
        rows, cols = self.grid.shape
        
        # Mark all seed locations
        # If value is 0, set regardless of current value (clear/reset)
        # Otherwise, only set if cell is currently unset (0)
        for seed_row, seed_col in locations:
            if 0 <= seed_row < rows and 0 <= seed_col < cols:
                if value == 0 or result[seed_row, seed_col] == 0:
                    result[seed_row, seed_col] = value
        
        # BFS from each seed to fill neighborhoods
        visited = np.zeros_like(self.grid, dtype=bool)
        queue = deque()
        
        # Initialize queue with all seeds at distance 0
        for seed_row, seed_col in locations:
            if 0 <= seed_row < rows and 0 <= seed_col < cols:
                visited[seed_row, seed_col] = True
                queue.append((seed_row, seed_col, 0))
        
        # BFS to mark all cells within max_distance
        while queue:
            r, c, dist = queue.popleft()
            
            # Skip if distance exceeds max
            if dist >= L:
                continue
            
            # Add neighbors (Manhattan neighbors: up, down, left, right)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                # Check bounds and visited
                if not (0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]):
                    continue
                
                # If value is 0, set regardless of current value (clear/reset)
                # Otherwise, only set if cell is currently unset (0)
                if value == 0 or result[nr, nc] == 0:
                    visited[nr, nc] = True
                    result[nr, nc] = value
                    queue.append((nr, nc, dist + 1))
        
        self.grid = result
        return self
    
    def count_positive_valued_cells(self):
        """Count positive-valued cells in the grid using hardware acceleration."""
        import numpy as np
        
        # Convert to numpy array
        np_array = np.asarray(self.grid)
        
        # Declare global at function level to avoid syntax errors
        global _accelerator_message_printed
        
        # --- Priority 1: Attempt JAX (GPU/TPU) ---
        try:
            import jax.numpy as jnp
            from jax import device_count as jax_device_count
            
            if jax_device_count() > 0:
                if not _accelerator_message_printed:
                    print("Using JAX with hardware acceleration...")
                    _accelerator_message_printed = True
                jnp_array = jnp.asarray(self.grid)
                # Count only positive values (> 0)
                count = int(jnp.sum(jnp_array > 0))
                return count
        except (ImportError, Exception) as e:
            pass  # JAX not available or error occurred
        
        # --- Priority 2: Attempt PyTorch (GPU) ---
        try:
            import torch
            
            # Try CUDA first
            if torch.cuda.is_available():
                if not _accelerator_message_printed:
                    print("Using PyTorch with CUDA GPU acceleration...")
                    _accelerator_message_printed = True
                pt_tensor = torch.tensor(self.grid, device='cuda')
                # Count only positive values (> 0)
                count = int(torch.sum(pt_tensor > 0).item())
                return count
            
            # Try MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if not _accelerator_message_printed:
                    print("Using PyTorch with MPS (Apple Silicon GPU) acceleration...")
                    _accelerator_message_printed = True
                pt_tensor = torch.tensor(self.grid, device='mps')
                # Count only positive values (> 0)
                count = int(torch.sum(pt_tensor > 0).item())
                return count
        except (ImportError, Exception) as e:
            pass  # PyTorch not available or error occurred
        
        # --- Fallback: NumPy (CPU) ---
        if not _accelerator_message_printed:
            print("Using NumPy on the CPU...")
            _accelerator_message_printed = True
        # Count only positive values (> 0)
        count = int(np.sum(np_array > 0))
        return count
    
    
    def __getitem__(self, index):
        return self.grid[index]
    
    def __setitem__(self, index, value):
        self.grid[index] = value
    
    @property
    def shape(self):
        return self.grid.shape
    
    def __repr__(self):
        return f"DenseGrid(shape={self.shape})"


class SparseGrid:
    """A sparse grid represented as a tuple (rows, cols, locations, values, L)."""
    
    def __init__(self, height, width):
        """
        Create a SparseGrid.
        
        Args:
            height: Number of rows (must be > 0)
            width: Number of columns (must be > 0)
        """
        if height <= 0 or width <= 0:
            raise ValueError(f"Grid dimensions must be > 0, got {height}x{width}")
        
        self.num_rows = height
        self.num_cols = width
        self.locations = []
        self.values = []
        self.L = 0
    
    def set_locations_to_value(self, locations, value=2):
        """Set locations and their values in the sparse grid."""
        self.locations = locations
        self.values = [value] * len(locations) if locations else []
        return self
    
    def set_neighborhoods_to_value(self, locations, L, value=2):
        """Set neighborhoods around locations and update L value."""
        self.locations = locations
        self.L = L
        self.values = [value] * len(locations) if locations else []
        return self
    
    def count_positive_valued_cells(self):
        """Count positive-valued cells in the sparse grid without creating grid."""
        positive_cells = set()
        
        # For each seed, enumerate all cells within Manhattan distance L
        for seed_row, seed_col in self.locations:
            # Enumerate cells in a Manhattan diamond around the seed
            for dr in range(-self.L, self.L + 1):
                for dc in range(-self.L, self.L + 1):
                    # Check Manhattan distance
                    manhattan_dist = abs(dr) + abs(dc)
                    if manhattan_dist <= self.L:
                        r = seed_row + dr
                        c = seed_col + dc
                        
                        # Check bounds
                        if 0 <= r < self.num_rows and 0 <= c < self.num_cols:
                            positive_cells.add((r, c))
        
        return len(positive_cells)
    
    def __repr__(self):
        return f"SparseGrid(rows={self.num_rows}, cols={self.num_cols}, locations={len(self.locations)}, L={self.L})"


def detect_available_accelerators():
    """
    Detect which hardware accelerators are available.
    
    Returns:
        dict: Dictionary containing information about available accelerators
    """
    accelerators = {
        'numpy': True,  # NumPy is assumed to be available
        'jax_gpu': False,
        'jax_tpu': False,
        'pytorch_cuda': False,
        'pytorch_mps': False,  # Apple Silicon GPU
    }
    
    # Check for JAX
    try:
        import jax
        from jax import device_count, local_devices
        
        devices = local_devices()
        for device in devices:
            if device.device_kind == 'gpu':
                accelerators['jax_gpu'] = True
            elif device.device_kind == 'tpu':
                accelerators['jax_tpu'] = True
                
        from jax import device_count as jax_device_count
        print(f"JAX is available with {jax_device_count()} device(s)")
    except ImportError:
        pass  # JAX not installed
    
    # Check for PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            accelerators['pytorch_cuda'] = True
            print(f"PyTorch CUDA: {torch.cuda.device_count()} GPU(s) available")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            accelerators['pytorch_mps'] = True
            print("PyTorch MPS (Apple Silicon) available")
    except ImportError:
        pass  # PyTorch not installed
    
    return accelerators

