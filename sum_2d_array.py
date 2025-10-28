"""
Efficient 2D array positive value counting with automatic hardware acceleration detection.

This module provides functions that automatically detect and use the fastest
available hardware (GPU via JAX/PyTorch, or optimized CPU via NumPy) to
efficiently count positive values (> 0) in 2D arrays.

This module is optimized for the use case: counting positive int elements
in a 2D array with H rows by W cols, where H and W are > 0 and element values are >= 0.
"""

import numpy as np


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
                
        print(f"JAX is available with {device_count()} device(s)")
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




def set_locations_to_one(shape_or_array, locations):
    """
    Create or modify a 2D array by setting specified locations to 2.
    
    This function efficiently sets multiple locations to 2 in a 2D array,
    useful for grid-based algorithms, sparse matrix operations, etc.
    
    Args:
        shape_or_array: Either a tuple (height, width) to create a new array,
                        or an existing 2D numpy array to modify
        locations: List of tuples (row, col) indicating positions to set to 2.
                  Can handle out-of-bounds indices (will be ignored).
    
    Returns:
        numpy.ndarray: Array with 2s at specified locations, 0s elsewhere
    
    Examples:
        >>> result = set_locations_to_one((5, 5), [(0, 0), (2, 2), (4, 4)])
        >>> # Creates 5x5 array with 2s at (0,0), (2,2), (4,4)
        
        >>> arr = np.zeros((3, 3))
        >>> result = set_locations_to_one(arr, [(1, 1), (1, 2)])
        >>> # Sets arr[1,1] and arr[1,2] to 2
    """
    import numpy as np
    
    # Determine if input is a shape tuple or an existing array
    if isinstance(shape_or_array, tuple):
        # Create new array filled with zeros
        height, width = shape_or_array
        array = np.zeros((height, width), dtype=np.int32)
        shape = (height, width)
    else:
        # Use existing array
        array = np.asarray(shape_or_array).copy()  # Copy to avoid modifying input
        height, width = array.shape
        shape = (height, width)
    
    # Handle empty locations list
    if not locations:
        return array
    
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
    valid_mask = ((locations[:, 0] >= 0) & (locations[:, 0] < height) &
                  (locations[:, 1] >= 0) & (locations[:, 1] < width))
    valid_locations = locations[valid_mask]
    
    if len(valid_locations) == 0:
        return array
    
    # Use advanced indexing to set values efficiently
    rows = valid_locations[:, 0]
    cols = valid_locations[:, 1]
    array[rows, cols] = 2
    
    return array


def create_sparse_grid(height, width, locations):
    """
    Convenience function: Create a new grid with specified locations set to 2.
    
    Args:
        height (int): Number of rows in the grid
        width (int): Number of columns in the grid
        locations: List of tuples (row, col) indicating positions to set to 2
    
    Returns:
        numpy.ndarray: Grid with 2s at specified locations, 0s elsewhere
    """
    return set_locations_to_one((height, width), locations)


def count_nonzero_2d_array(input_array):
    """
    Count positive values (> 0) in a 2D array using the fastest available hardware.
    
    This is the PRIMARY function for counting positive elements in a 2D array,
    automatically selecting the best hardware (JAX/TPU, PyTorch/GPU, or NumPy/CPU).
    
    This function counts only positive values (greater than 0), excluding:
    - Zero values
    - Negative values
    
    This is the recommended function for most use cases where you need to count
    positive values in a 2D array (H x W where H, W > 0 and elements >= 0).
    
    Args:
        input_array (list of lists or numpy.ndarray): The 2D array to count positive values
    
    Returns:
        int: Number of positive elements (> 0) in the array
    
    Examples:
        >>> array = np.array([[1, 0, 2], [0, -1, 3], [4, 0, 5]])
        >>> count = count_nonzero_2d_array(array)
        >>> # Returns 5 (positive elements: 1, 2, 3, 4, 5)
    """
    import numpy as np
    
    # Convert to numpy array
    np_array = np.asarray(input_array)
    
    # --- Priority 1: Attempt JAX (GPU/TPU) ---
    try:
        import jax.numpy as jnp
        from jax import device_count
        
        if device_count() > 0:
            print("Using JAX with hardware acceleration...")
            jnp_array = jnp.asarray(input_array)
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
            print("Using PyTorch with CUDA GPU acceleration...")
            pt_tensor = torch.tensor(input_array, device='cuda')
            # Count only positive values (> 0)
            count = int(torch.sum(pt_tensor > 0).item())
            return count
        
        # Try MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Using PyTorch with MPS (Apple Silicon GPU) acceleration...")
            pt_tensor = torch.tensor(input_array, device='mps')
            # Count only positive values (> 0)
            count = int(torch.sum(pt_tensor > 0).item())
            return count
    except (ImportError, Exception) as e:
        pass  # PyTorch not available or error occurred
    
    # --- Fallback: NumPy (CPU) ---
    print("Using NumPy on the CPU...")
    # Count only positive values (> 0)
    count = int(np.sum(np_array > 0))
    return count


# --- Example Usage ---

if __name__ == "__main__":
    # Detect available accelerators
    print("Detecting hardware accelerators...")
    accelerators = detect_available_accelerators()
    print("\nAvailable accelerators:", accelerators)
    print()
    
    # Example 1: Count positive values (PRIMARY OPERATION)
    print("Example 1: Counting positive values (> 0)")
    test_array = np.array([[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 0, 0]])
    print("Input array:\n", test_array)
    positive_count = count_nonzero_2d_array(test_array)
    print(f"Positive elements: {positive_count}")
    print()
    
    # Example 2: Set locations to 2 and count
    print("Example 2: Setting locations to 2 and counting positive values")
    locations = [(0, 0), (1, 2), (2, 4), (3, 1), (4, 3)]
    grid = set_locations_to_one((5, 5), locations)
    print(f"Grid:\n{grid}")
    print(f"Positive count: {count_nonzero_2d_array(grid)}")
    print()
    
    # Example 3: Complete workflow - set and count
    print("Example 3: Complete workflow - set locations to 2 and count")
    grid = set_locations_to_one((5, 5), [(0, 0), (2, 2), (4, 4)])
    print("Grid with locations set to 2:\n", grid)
    print(f"Positive cells: {count_nonzero_2d_array(grid)}")
    print()
    
    # Example 4: Count from standard Python list
    print("Example 4: Count positive from Python list")
    large_list = [[i * j for j in range(100)] for i in range(100)]
    count = count_nonzero_2d_array(large_list)
    print(f"Positive count from list: {count:,}")
    print()
    
    # Example 5: NumPy array operations
    print("Example 5: NumPy array operations")
    large_numpy_array = np.random.rand(100, 100)
    count = count_nonzero_2d_array(large_numpy_array)
    print(f"Positive count: {count}")
    print()
    
    # Example 6: Large array performance
    print("Example 6: Large array performance (1000x1000)")
    huge_array = np.random.rand(1000, 1000) * 100
    count = count_nonzero_2d_array(huge_array)
    print(f"Positive count: {count:,}")
    print()
    
    # Example 7: Modify existing array
    print("Example 7: Modifying existing array")
    existing = np.zeros((3, 3))
    new_locations = [(0, 1), (1, 1), (2, 1)]
    modified = set_locations_to_one(existing, new_locations)
    print("Modified array:\n", modified)
    print(f"Positive count: {count_nonzero_2d_array(modified)}")
    print()
    
    # Example 8: Mixed values - counts only positive
    print("Example 8: Counting with mixed positive, negative, and zero values")
    mixed_array = np.array([[1, -2, 0, 4], [0, -1, 3, 0], [-5, 0, 2, 0]])
    print("Input array:\n", mixed_array)
    count = count_nonzero_2d_array(mixed_array)
    print(f"Positive count (only > 0): {count}")
    print("Note: Negative values and zeros are excluded")

