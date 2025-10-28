# Efficient 2D Array Positive Value Counting with Hardware Acceleration

This project provides highly optimized Python functions for counting positive values in 2D arrays, automatically detecting and using the fastest available hardware accelerators (GPU/TPU via JAX or PyTorch, or optimized CPU via NumPy).

## Features

- **Automatic hardware detection**: Automatically detects available GPU/TPU hardware
- **Intelligent fallback**: Uses JAX → PyTorch → NumPy in priority order
- **Flexible input**: Accepts Python lists of lists or NumPy arrays
- **Optional filtering**: Can sum all elements or only positive elements
- **Efficient sparse operations**: Set multiple locations to 1 with vectorized operations
- **Zero external dependencies (minimum)**: Works with just NumPy

## Installation

### Basic Installation (CPU only - NumPy)
```bash
pip install numpy
```

### With GPU Support (PyTorch)
```bash
# For CUDA GPUs
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon GPUs
pip install torch
```

### With GPU Support (JAX)
```bash
# For CUDA GPUs
pip install jax[cuda11_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Or for the latest CPU version
pip install jax jaxlib
```

### Install All Options
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from count_2d_array import intelligent_sum_2d_array, sum_only_positive, set_locations_to_one, detect_available_accelerators

# Detect available hardware
accelerators = detect_available_accelerators()

# Sum all elements of a 2D array
my_2d_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
total = intelligent_sum_2d_array(my_2d_array)
print(f"Total sum: {total}")

# Sum only positive elements
array_with_negatives = [[1, -2, 3], [0, 5, -1], [2, 0, 1]]
positive_total = intelligent_sum_2d_array(array_with_negatives, positive_only=True)
print(f"Sum of positive elements: {positive_total}")

# Set specific locations to 1
grid = set_locations_to_one((5, 5), [(0, 0), (1, 1), (2, 2)])
print(f"Grid:\n{grid}")
```

## Functions

### `intelligent_sum_2d_array(input_array, positive_only=False)`
Main function that automatically uses the fastest available hardware.

**Parameters:**
- `input_array`: 2D array (list of lists or numpy.ndarray)
- `positive_only`: If True, only sum elements > 0

**Returns:**
- Sum of elements (or positive elements if `positive_only=True`)

### `sum_only_positive(input_array)`
Convenience wrapper that sums only positive elements.

### `set_locations_to_one(shape_or_array, locations)`
Efficiently sets specified locations in a 2D array to 1. Can create a new array or modify an existing one.

**Parameters:**
- `shape_or_array`: Either a tuple (height, width) to create a new array, or an existing 2D numpy array
- `locations`: List of tuples (row, col) indicating positions to set to 1

**Returns:**
- numpy.ndarray with 1s at specified locations, 0s elsewhere

**Example:**
```python
# Create new grid with locations set to 1
grid = set_locations_to_one((5, 5), [(0, 0), (2, 2), (4, 4)])

# Modify existing array
existing = np.zeros((3, 3))
modified = set_locations_to_one(existing, [(1, 1)])
```

### `create_sparse_grid(height, width, locations)`
Convenience function to create a new grid with specified locations set to 1.

**Parameters:**
- `height`: Number of rows
- `width`: Number of columns
- `locations`: List of tuples (row, col) indicating positions to set to 1

**Returns:**
- numpy.ndarray grid with 1s at specified locations

### `detect_available_accelerators()`
Detects which hardware accelerators are available.

**Returns:**
- Dictionary with acceleration capabilities

## Performance

The function uses hardware acceleration in this priority order:

1. **JAX (GPU/TPU)** - Fastest for large arrays on compatible hardware
2. **PyTorch (GPU)** - Excellent CUDA/MPS support
3. **NumPy (CPU)** - Highly optimized C-based implementation

All methods are significantly faster than naive Python loops.

## Example Usage

### Summing Arrays

```python
import numpy as np
from count_2d_array import intelligent_sum_2d_array

# Create a large random array
large_array = np.random.rand(1000, 1000)

# Sum all elements
total = intelligent_sum_2d_array(large_array)
print(f"Total: {total:.2f}")

# Sum only positive elements
positive_total = intelligent_sum_2d_array(large_array, positive_only=True)
print(f"Positive sum: {positive_total:.2f}")
```

### Setting Locations

```python
from count_2d_array import set_locations_to_one, create_sparse_grid

# Create a grid with specific locations set to 1
locations = [(0, 0), (1, 1), (2, 2)]
grid = set_locations_to_one((5, 5), locations)
print(grid)
# Output:
# [[1 0 0 0 0]
#  [0 1 0 0 0]
#  [0 0 1 0 0]
#  [0 0 0 0 0]
#  [0 0 0 0 0]]

# Or use the convenience function
grid = create_sparse_grid(3, 3, [(0, 0), (2, 2)])
```

### Complete Workflow

```python
from count_2d_array import set_locations_to_one, intelligent_sum_2d_array

# Create a sparse grid
active_locations = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]
grid = set_locations_to_one((5, 5), active_locations)

# Count active cells
active_count = intelligent_sum_2d_array(grid)
print(f"Active cells: {active_count}")
```

## Performance Tips

1. **Install GPU libraries**: For arrays larger than 1000x1000, GPU acceleration provides significant speedup
2. **Use NumPy arrays**: Converting Python lists to NumPy arrays adds overhead
3. **Batch operations**: For multiple arrays, consider batching operations

## Requirements

- Python 3.7+
- NumPy 1.20+ (required)
- PyTorch 2.0+ (optional, for GPU support)
- JAX 0.4+ (optional, for GPU/TPU support)

## License

This code is provided as-is for demonstration purposes.

