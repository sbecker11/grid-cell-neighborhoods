# Efficient 2D Array Positive Value Counting with Hardware Acceleration

This project provides highly optimized Python functions for counting positive values in 2D arrays, automatically detecting and using the fastest available hardware accelerators (GPU/TPU via JAX or PyTorch, or optimized CPU via NumPy).

## Files

- **`grid_counting.py`** - Main module with all counting and Manhattan neighborhood functions
- **`grid_counting_tests.py`** - Comprehensive test suite including functionality tests and performance benchmarks
- **`requirements.txt`** - Python package dependencies
- **`Counting_grid-cell_neighborhoods.pdf`** - Project description and detailed documentation
- **`README.md`** - This documentation

## Features

- **Automatic hardware detection**: Automatically detects available GPU/TPU hardware
- **Intelligent fallback**: Uses JAX → PyTorch → NumPy in priority order
- **Flexible input**: Accepts Python lists of lists or NumPy arrays
- **Efficient sparse operations**: Set multiple locations to any value with vectorized operations
- **Manhattan neighborhood operations**: Fill neighborhoods around seed points with efficient BFS
- **Direct counting optimization**: Count positive-valued cells in neighborhoods without grid allocation (3-6x faster)
- **Zero external dependencies (minimum)**: Works with just NumPy

## Installation

### Setup Virtual Environment (Recommended)

```bash
# Create virtual environment, activate, and install dependencies
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### Required Libraries

**Minimum (CPU only):**
- `numpy` (>=1.20.0, <2.0.0)

**For PyTorch GPU acceleration:**
- `torch` (with CUDA or MPS support)

**For JAX GPU/TPU acceleration:**
- `jax`
- `jaxlib`

## Examples

```python
from grid_counting import count_positive_valued_cells_in_2d_array, set_locations_to_value_full_grid, count_manhattan_neighborhoods_sparse_grid

# Count positive values in a 2D array
my_2d_array = np.array([
    [1, 2, 0],   # row 0
    [0, 5, -3],  # row 1
    [4, 0, 1]    # row 2
])
count = count_positive_valued_cells_in_2d_array(my_2d_array)  # Returns 5

# Set specific locations to 2
num_rows, num_cols = 5, 5
grid = set_locations_to_value_full_grid((num_rows, num_cols), [(0, 0), (1, 1), (2, 2)], value=2)

# Count Manhattan neighborhoods (OPTIMAL - no grid creation)
count = count_manhattan_neighborhoods(
    num_rows=100, num_cols=100,
    seeds=[(10, 10), (20, 20), (30, 30)],
    L=3  # Manhattan distance
)
# Returns count of positive-valued cells within L=3 from all seeds
```

## Functions

### `count_positive_valued_cells_in_2d_array(array)`
Counts positive values in a 2D array using the fastest available hardware.

**Parameters:**
- `array`: 2D numpy array

**Returns:**
- Count of elements > 0

### `set_locations_to_value_full_grid(shape_or_array, locations, value=2)`
Efficiently sets specified locations in a 2D array to a value.

**Parameters:**
- `shape_or_array`: Either a tuple (height, width) to create a new array, or an existing 2D numpy array
- `locations`: List of tuples (row, col) indicating positions to set
- `value`: The value to set at the specified locations (default=2)

**Returns:**
- numpy.ndarray with specified values at locations, 0 elsewhere

**Example:**
```python
import numpy as np

# Create new grid with locations set to 2
grid = set_locations_to_value_full_grid((5, 5), [(0, 0), (2, 2), (4, 4)], value=2)

# Set locations to a different value
grid = set_locations_to_value_full_grid(grid, [(1, 1), (3, 3)], value=5)
```

### `count_manhattan_neighborhoods_sparse_grid(num_rows, num_cols, seeds, L)`
**⚡ RECOMMENDED FUNCTION** - Count positive-valued cells within Manhattan distance L from seed locations.

This is the optimal top-level function for counting without creating a grid. It's 3-6x faster and uses no grid memory allocation.

**Parameters:**
- `num_rows`: Number of rows in the grid
- `num_cols`: Number of columns in the grid
- `seeds`: List of tuples (row, col) - seed locations
- `L`: Maximum Manhattan distance from seeds to count

**Returns:**
- int: Count of positive-valued cells within Manhattan distance L from all seeds

**Benefits:**
- Fastest approach (3-6x faster than creating grid)
- Memory efficient (no grid allocation)
- Handles overlapping neighborhoods correctly
- Does not count neighborhood cells that lie outside of array boundary

**Example:**
```python
from grid_counting import count_manhattan_neighborhoods

# Count neighborhoods around 3 seeds
seeds = [(10, 10), (50, 50), (90, 90)]
count = count_manhattan_neighborhoods(
    num_rows=100, num_cols=100,
    seeds=seeds, 
    L=5
)
# Returns count of positive-valued cells within L=5 from all seeds
```

### `set_manhattan_neighborhoods_full_grid(grid, seeds, max_distance, target_value=2)`
Sets values in a 2D array for cells within Manhattan distance from seed points using BFS.

**Parameters:**
- `grid`: 2D numpy array (will be modified)
- `seeds`: List of tuples (row, col) - seed positions
- `max_distance`: Maximum Manhattan distance to fill
- `target_value`: Value to set (default=2)

**Returns:**
- Modified grid with filled neighborhoods

### `create_sparse_grid_full_grid(height, width, locations)`
Convenience function to create a new grid with specified locations set to 2.

**Parameters:**
- `height`: Number of rows
- `width`: Number of columns
- `locations`: List of tuples (row, col) indicating positions to set to 2

**Returns:**
- numpy.ndarray grid with 2 at specified locations

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

### Performance Comparison

For neighborhood counting operations:

- **Direct counting** (`count_manhattan_neighborhoods_sparse_grid`): 3-6x faster than creating a grid
- **BFS approach** (`set_manhattan_neighborhoods_full_grid`): Optimal for sparse seeds, ~0.06ms for 2 seeds
- **Vectorized approach**: Best for dense seeds or large grids

### Performance Testing

The test suite includes performance benchmarks that automatically run:

1. **Hardware Detection** - Verifies CPU/GPU/TPU accelerator detection
2. **Small Array Performance** - Benchmarks counting operations on small arrays
3. **Manhattan Performance** - Compares direct counting vs grid creation (3-6x speedup)

Run `python grid_counting_tests.py` to see all functionality and performance tests.

## Optimal Approaches for Setting Manhattan Neighborhood Values

### **What is BFS? (Breadth-First Search)**

BFS is a graph traversal algorithm that explores all neighbors at the current depth before moving to the next level. For Manhattan neighborhoods:

1. **Starts at each seed location** - Beginning points for filling
2. **Expands outward level by level** - Processes all cells at distance 1, then 2, then 3, etc.
3. **Uses a queue (FIFO)** - First cells found are processed first
4. **Marks visited cells** - Prevents revisiting cells or double-counting overlaps
5. **Stops at max distance** - Only fills up to the specified Manhattan distance L

**Manhattan Distance Formula:** `|row₁ - row₂| + |col₁ - col₂|`

For a seed at (5, 5) with L=3, BFS would fill a diamond-shaped pattern of all cells within 3 steps in the four cardinal directions (up, down, left, right).

### **Current Implementation (BFS - Recommended)**

The `set_manhattan_neighborhoods_full_grid()` function uses Breadth-First Search (BFS), which is **optimal for most cases**.

**Why BFS is optimal:**
1. ✅ **Handles overlaps correctly** - Visited cells are never revisited
2. ✅ **Memory efficient** - Only stores frontier, not all distances
3. ✅ **Fastest for sparse seeds** - O(k * L²) where k = number of seeds
4. ✅ **Early termination** - Stops at max_distance

**Performance results:**
- Sparse seeds (2 seeds): ~0.06ms
- Medium seeds (5 seeds): ~0.39ms  
- Dense seeds (10 seeds): ~0.47ms

### **Alternative Approaches**

**1. Vectorized Distance Computation**
```python
# For each seed, compute Manhattan distance to ALL cells
for seed_row, seed_col in seeds:
    manhattan_dist = np.abs(row_coords - seed_row) + np.abs(col_coords - seed_col)
    result[manhattan_dist <= max_distance] = target_value
```

**Best for:** Dense seeds, large grids, when you need all distance values  
**Time:** O(k * n * m)  
**Space:** O(k * n * m)

**2. Convolution-Based**
```python
from scipy import signal
# Create mask and apply convolution
mask = create_distance_mask(L)
convolved = signal.convolve2d(seed_mask, mask, mode='same')
```

**Best for:** Repeated operations on same grid structure  
**Time:** O(n * m * L²)  
**Notes:** Requires scipy, best when processing many grids

**3. Morphological Dilation**
```python
from scipy.ndimage import binary_dilation
# Dilate seeds L times
dilated = binary_dilation(seed_mask, iterations=L)
```

**Best for:** Very large grids, morphological operations  
**Notes:** Uses scipy, can be faster for L=1

### **Performance Comparison Results**

| Approach | Array Size | 2 Seeds (L=3) | 5 Seeds (L=5) | 10 Seeds (L=4) |
|:---------|:----------:|:-------------:|:-------------:|:--------------:|
| **BFS (Current)** | 100×100 | 0.06ms | 0.39ms | 0.47ms |
| **Vectorized** | 100×100 | 0.06ms | 0.19ms | 1.12ms |
| **Convolution** | 100×100 | 0.66ms | 5.26ms | 34.67ms |
| **Morphological** | 100×100 | 0.08ms | 0.42ms | 0.52ms |

### **Recommendations**

**Use BFS (Current) when:**
- ✅ Sparse or medium density seeds
- ✅ Overlapping neighborhoods need correct handling
- ✅ Memory efficiency matters
- ✅ Small to medium L values (≤ 10)

**Use Vectorized when:**
- ✅ Many seeds (>20)
- ✅ You need distance matrices anyway
- ✅ Large grids (1000×1000+)
- ✅ Already have coordinate arrays

**Use Morphological when:**
- ✅ Very large grids
- ✅ L = 1 (4-neighborhood)
- ✅ Already using scipy.ndimage
- ✅ Need other morphological operations

### **Why BFS is Optimal**

The current implementation in `set_manhattan_neighborhoods_full_grid()` uses BFS because:

1. **Automatic overlap handling** - Visited array prevents double-counting
2. **Early termination** - Only processes cells within distance L
3. **Memory efficient** - Queue size bounded by perimeter of neighborhood
4. **Simple logic** - Easy to understand and maintain
5. **Proven performance** - Fast for typical use cases

**Complexity:**
- Time: O(k * L²) per seed, where L² is the area of the diamond
- Space: O(n * m) for visited array
- Actually visits only O(k * L²) cells, not O(n * m)

**Example for L=3:**
- One seed creates ~13 cells (1 + 4 + 8 diamond)
- BFS visits exactly these 13 cells
- Vectorized computes distance for all n×m cells
- BFS is faster when n×m >> L²

## Example Usage

### Counting Positive Values

```python
import numpy as np
from grid_counting import count_positive_valued_cells_in_2d_array

# Create a large random array
large_array = np.random.rand(1000, 1000)

# Count positive elements
count = count_positive_valued_cells_in_2d_array(large_array)
print(f"Positive count: {count}")
```

### Setting Locations

```python
from grid_counting import set_locations_to_value_full_grid, create_sparse_grid_full_grid

# Create a grid with specific locations set to 2
locations = [(0, 0), (1, 1), (2, 2)]
grid = set_locations_to_value_full_grid((5, 5), locations)
print(grid)
# Output:
# [[2 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 2 0 0]
#  [0 0 0 0 0]
#  [0 0 0 0 0]]

# Or use the convenience function
grid = create_sparse_grid_full_grid(3, 3, [(0, 0), (2, 2)])
```

### Manhattan Neighborhoods

```python
from grid_counting import count_manhattan_neighborhoods

# Count without creating grid (OPTIMAL)
count = count_manhattan_neighborhoods(
    num_rows=100, num_cols=100,
    seeds=[(10, 10), (20, 20)],
    L=3
)
print(f"Cells in neighborhoods: {count}")

# For visualization with grid
from grid_counting import set_manhattan_neighborhoods_full_grid

grid = np.zeros((100, 100))
seeds = [(10, 10), (20, 20)]
set_manhattan_neighborhoods_full_grid(grid, seeds, max_distance=3, target_value=2)
```

### Complete Workflow

```python
from grid_counting import set_locations_to_value_full_grid, count_positive_valued_cells_in_2d_array, count_manhattan_neighborhoods_sparse_grid

# Option 1: Count with grid creation
active_locations = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]
grid = set_locations_to_value((5, 5), active_locations)
count = count_positive_valued_cells_in_2d_array(grid)
print(f"Active cells: {count}")

# Option 2: Count without grid (FASTER!)
count = count_manhattan_neighborhoods(
    num_rows=5, num_cols=5,
    seeds=active_locations,
    L=0  # Exact locations only
)
print(f"Active cells: {count}")
```

## Performance Tips

1. **Install GPU libraries**: For arrays larger than 1000×1000, GPU acceleration provides significant speedup
2. **Use NumPy arrays**: Converting Python lists to NumPy arrays adds overhead
3. **Use `count_manhattan_neighborhoods_sparse_grid`**: For counting only, don't create grids (3-6x faster)
4. **Batch operations**: For multiple arrays, consider batching operations

## Requirements

- Python 3.7+
- NumPy 1.20+ (required)
- PyTorch 2.0+ (optional, for GPU support)
- JAX 0.4+ (optional, for GPU/TPU support)

## Running Tests

```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run all tests (functionality + performance benchmarks)
python grid_counting_tests.py

# Run with pytest (if installed)
pytest grid_counting_tests.py -v
```

## License

This code is provided as-is for demonstration purposes.