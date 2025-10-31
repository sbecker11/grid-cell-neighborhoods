# Efficient 2D Array Positive Value Counting with Hardware Acceleration

This project provides highly optimized Python functions for counting positive values in 2D arrays, automatically detecting and using the fastest available hardware accelerators (GPU/TPU via JAX or PyTorch, or optimized CPU via NumPy).

## 🌐 Interactive Test Runner

**Try it in your browser:** [https://sbecker11.github.io/grid-cell-neighborhoods/](https://sbecker11.github.io/grid-cell-neighborhoods/)

Run all unit tests interactively in your browser using Pyodide (Python compiled to WebAssembly). No installation required!

## Files

- **`grid_counting.py`** - Main module with all counting and Manhattan neighborhood functions
- **`grid_counting_tests.py`** - Comprehensive test suite including functionality tests and performance benchmarks
- **`requirements.txt`** - Python package dependencies
- **`Counting_grid-cell_neighborhoods.pdf`** - Project description (PDF version)
- **`Counting_grid-cell_neighborhoods.txt`** - Project description (text version)
- **`README.md`** - This documentation

## Features

- **Automatic hardware detection**: Automatically detects available GPU/TPU hardware
- **Intelligent fallback**: Uses JAX → PyTorch → NumPy in priority order
- **Flexible input**: Accepts Python lists of lists or NumPy arrays
- **Efficient sparse operations**: Set multiple locations to any value with vectorized operations
- **Manhattan neighborhood operations**: Fill neighborhoods around seed points with efficient BFS
- **Direct counting optimization**: Count positive-valued cells in neighborhoods without grid allocation (3-6x faster)
- **Zero external dependencies (minimum)**: Works with just NumPy

## Dense vs Sparse Grids

This library provides two approaches for working with 2D grids:

### **Dense Arrays (`DenseGrid`)**
A **dense grid** is a fully allocated 2D NumPy array where every cell exists in memory, even if most cells are zero or empty.

**Characteristics:**
- ✅ Stores complete grid in memory (all rows × columns cells)
- ✅ Fast random access to any cell
- ✅ Supports visualization and full grid operations
- ❌ Uses O(rows × cols) memory regardless of how many cells are active
- ❌ Memory-intensive for large, mostly-empty grids

**Use when:**
- You need to visualize or modify the full grid
- You'll access cells randomly throughout the grid
- The grid is relatively small or mostly filled
- You need the actual grid values for further processing

**Example:**
```python
from grid_counting import DenseGrid

# Allocates full 100×100 = 10,000 cells in memory
denseGrid = DenseGrid(100, 100)
denseGrid.set_locations_to_value([(10, 10), (20, 20)], value=2)
count = denseGrid.count_positive_valued_cells()  # Counts from full grid array
```

### **Sparse Arrays (`SparseGrid`)**
A **sparse grid** stores only the locations and values of active cells without allocating the full grid.

**Characteristics:**
- ✅ Memory efficient: Only stores active cell locations
- ✅ 3-6x faster for counting operations (no grid allocation)
- ✅ Optimal for large grids with few active cells
- ❌ Cannot access arbitrary cells directly (no full grid)
- ❌ Cannot visualize full grid (only has location data)

**Use when:**
- You only need counts, not the full grid
- Grid is large but most cells are inactive/zero
- Memory is constrained
- Performance is critical for counting operations

**Example:**
```python
from grid_counting import SparseGrid

# No grid allocation - only stores locations [(10,10), (20,20)]
sparseGrid = SparseGrid(100, 100)
sparseGrid.set_neighborhoods_to_value([(10, 10), (20, 20)], L=3, value=2)
count = sparseGrid.count_positive_valued_cells()
```

### **Quick Decision Guide**

| Need | Use |
|------|-----|
| Full grid for visualization | `DenseGrid` |
| Full grid for further processing | `DenseGrid` |
| Random cell access | `DenseGrid` |
| Just counting cells | `SparseGrid` |
| Large mostly-empty grids | `SparseGrid` |
| Maximum performance for counting | `SparseGrid` |

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
import numpy as np
from grid_counting import DenseGrid, SparseGrid

# Create a dense grid and count positive values
my_2d_array = np.array([
    [1, 2, 0],   # row 0
    [0, 5, -3],  # row 1
    [4, 0, 1]    # row 2
])
denseGrid = DenseGrid(my_2d_array)
count = denseGrid.count_positive_valued_cells()  # Returns 5

# Set specific locations to 2
denseGrid = DenseGrid(5, 5)
denseGrid.set_locations_to_value([(0, 0), (1, 1), (2, 2)], value=2)

# Count Manhattan neighborhoods (OPTIMAL - no grid creation)
sparseGrid = SparseGrid(100, 100)
sparseGrid.set_neighborhoods_to_value([(10, 10), (20, 20), (30, 30)], L=3, value=2)
count = sparseGrid.count_positive_valued_cells()
# Returns count of positive-valued cells within L=3 from all locations
```

## Classes

### `DenseGrid(height, width, initial_value=0)`
A dense 2D grid represented as a numpy array.

**Parameters:**
- `height`: Number of rows
- `width`: Number of columns
- `initial_value`: Value to fill grid with (default=0)

**Methods:**
- `set_locations_to_value(locations, value=2)`: Set specified locations to a value
- `set_neighborhoods_to_value(locations, L=3, value=2)`: Set neighborhoods around locations using BFS
- `count_positive_valued_cells()`: Count positive-valued cells using hardware acceleration

**Example:**
```python
import numpy as np
from grid_counting import DenseGrid

# Create new grid
denseGrid = DenseGrid(5, 5)

# Set locations to a value
denseGrid.set_locations_to_value([(0, 0), (2, 2), (4, 4)], value=2)

# Set neighborhoods around locations
denseGrid.set_neighborhoods_to_value([(1, 1), (3, 3)], L=2, value=2)

# Count positive values
count = denseGrid.count_positive_valued_cells()
```

### `SparseGrid(height, width)`
A sparse grid that stores only seed locations and Manhattan distance, avoiding full grid allocation.

**Parameters:**
- `height`: Number of rows (must be > 0)
- `width`: Number of columns (must be > 0)

**Methods:**
- `set_locations_to_value(locations, value=2)`: Set specified locations to a single value (applies same value to all locations)
- `set_neighborhoods_to_value(locations, L, value=2)`: Set neighborhoods around locations using BFS (applies same value to all unset cells)
- `count_positive_valued_cells()`: Count positive-valued cells without creating grid

#### Definition: locations
- A list or iterable of zero-based `(row, col)` integer tuples, e.g. `[(r, c), ...]`.
- Rows in `[0, height-1]`, columns in `[0, width-1]`; out-of-bounds entries are ignored.
- Duplicates are allowed and have no additional effect.
- Single location may be provided as a 2-tuple `(row, col)`; it is treated as one item.

**Benefits:**
- Fastest approach (3-6x faster than creating grid)
- Memory efficient (no grid allocation)
- Handles overlapping neighborhoods correctly
- Does not count neighborhood cells that lie outside of array boundary

**Example:**
```python
from grid_counting import SparseGrid

# Create sparse grid
sparseGrid = SparseGrid(100, 100)

# Set neighborhoods around locations
sparseGrid.set_neighborhoods_to_value([(10, 10), (50, 50), (90, 90)], L=5, value=2)

# Count positive-valued cells (no grid creation!)
count = sparseGrid.count_positive_valued_cells()
```

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

- **Direct counting** (`SparseGrid.count_positive_valued_cells()`): 3-6x faster than creating a grid
- **BFS approach** (`DenseGrid.set_neighborhoods_to_value()`): Optimal for sparse seeds, ~0.06ms for 2 seeds
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

The `DenseGrid.set_neighborhoods_to_value()` method uses Breadth-First Search (BFS), which is **optimal for most cases**.

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

The current implementation in `DenseGrid.set_neighborhoods_to_value()` uses BFS because:

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
from grid_counting import DenseGrid

# Create a large random array
large_array = np.random.rand(1000, 1000)

# Count positive elements
denseGrid = DenseGrid(large_array)
count = denseGrid.count_positive_valued_cells()
print(f"Positive count: {count}")
```

### Setting Locations

```python
from grid_counting import DenseGrid

# Create a grid with specific locations set to 2
locations = [(0, 0), (1, 1), (2, 2)]
denseGrid = DenseGrid(5, 5)
denseGrid.set_locations_to_value(locations, value=2)
print(denseGrid.grid)
# Output:
# [[2 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 2 0 0]
#  [0 0 0 0 0]
#  [0 0 0 0 0]]
```

### Manhattan Neighborhoods

```python
from grid_counting import SparseGrid, DenseGrid

# Count without creating grid (OPTIMAL)
sparseGrid = SparseGrid(100, 100)
sparseGrid.set_neighborhoods_to_value([(10, 10), (20, 20)], L=3, value=2)
count = sparseGrid.count_positive_valued_cells()
print(f"Cells in neighborhoods: {count}")

# For visualization with grid
denseGrid = DenseGrid(100, 100)
locations = [(10, 10), (20, 20)]
denseGrid.set_neighborhoods_to_value(locations, L=3, value=2)
```

### Complete Workflow

```python
from grid_counting import DenseGrid, SparseGrid

# Option 1: Count with grid creation
active_locations = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]
denseGrid = DenseGrid(5, 5)
denseGrid.set_locations_to_value(active_locations, value=2)
count = denseGrid.count_positive_valued_cells()
print(f"Active cells: {count}")

# Option 2: Count without grid (FASTER!)
sparseGrid = SparseGrid(5, 5)
sparseGrid.set_locations_to_value(active_locations, value=2)
count = sparseGrid.count_positive_valued_cells()
print(f"Active cells: {count}")
```

## Performance Tips

1. **Install GPU libraries**: For arrays larger than 1000×1000, GPU acceleration provides significant speedup
2. **Use NumPy arrays**: Converting Python lists to NumPy arrays adds overhead
3. **Use `SparseGrid`**: For counting only, don't create grids (3-6x faster)
4. **Batch operations**: For multiple arrays, consider batching operations

## Requirements

- Python 3.7+
- NumPy 1.20+ (required)
- PyTorch 2.0+ (optional, for GPU support)
- JAX 0.4+ (optional, for GPU/TPU support)

## Running Tests

This project includes comprehensive testing with 31 test cases covering functionality, edge cases, and performance benchmarks. Tests are run automatically on every commit via GitHub Actions across multiple operating systems.

### Test Coverage

The test suite (`grid_counting_tests.py`) includes:

1. **Basic functionality tests** (15 tests)
   - Setting locations to values
   - Setting Manhattan neighborhoods
   - Counting positive-valued cells
   - Edge cases (empty, out-of-bounds, single locations)

2. **Hardware detection tests** (1 test)
   - Detects available accelerators (NumPy, PyTorch, JAX)
   - Reports OS-specific information
   - Validates CPU/GPU/TPU detection

3. **Performance benchmarks** (1 test)
   - Small array performance validation
   - Ensures operations complete within expected time limits

4. **Project requirement examples** (4 tests)
   - Example scenarios from project specification
   - Validates correctness of neighborhood calculations

5. **Edge case tests** (8 tests)
   - Corner placements
   - Odd-shaped arrays (1×21, 10×1, 2×2, 1×1)
   - Extreme values (N=0, N >> max dimensions)
   - Error handling (zero dimensions)

### Local Testing

#### Prerequisites

Install dependencies:
```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Optional: For hardware detection tests (installs PyTorch and JAX)
pip install torch>=2.0.0 jax>=0.4.0 jaxlib>=0.4.0
```

#### Run All Tests

```bash
# Run all 31 test cases
python grid_counting_tests.py
```

Expected output:
- Each test shows numbered headers: `TEST N: [test name]`
- OS information displayed in first test
- Hardware detection results in hardware test
- Performance metrics in performance test
- Summary: `✓ All tests passed!` (if all succeed)

#### Run Specific Test Categories

```bash
# Using pytest (if installed)
pytest grid_counting_tests.py -v

# Run specific test function
pytest grid_counting_tests.py::test_set_dense_grid_basic -v
```

#### Test Output Format

Local test runs display numbered headers with consistent formatting:
```
======================================================================
TEST 1: set_dense_grid (diagonal pattern)
======================================================================
Operating System: Darwin 24.6.0 (...)
Platform: x86_64 / i386
...
✓ Basic set_locations test passed

======================================================================
TEST 2: set_dense_grid with empty locations (no locations specified)
======================================================================
...
```

**Formatting rules:**
- Numbered headers: `TEST N: [title]` wrapped with `====` lines (no leading blank line)
- Blank line appears after each test's pass message
- Unnumbered internal headers are automatically filtered out
- Browser output matches shell output format

### Browser-Based Testing

**No installation required!** Run tests directly in your browser:

1. Visit: [https://sbecker11.github.io/grid-cell-neighborhoods/](https://sbecker11.github.io/grid-cell-neighborhoods/)
2. Click "Run Tests Now" (after Pyodide loads)
3. Navigate through all 31 tests using pagination
4. View detailed output for each test case

The browser test runner:
- Loads Python runtime (Pyodide) automatically
- Displays tests with pagination (31 pages)
- Shows OS information and hardware detection
- Works on any device with a modern browser

### Continuous Integration (GitHub Actions)

Tests automatically run on every push and pull request via GitHub Actions:

#### Test Matrix

Tests run on **3 operating systems** with Python 3.11:

- **macOS** (macos-latest)
- **Windows** (windows-latest)
- **Linux** (ubuntu-latest)

![GitHub Actions Test Results](docs/GitHub%20Actions%20Tests%20(Automated%20Cross-Platform%20Testing)%20Results.png)

#### View CI Results

1. Go to your repository on GitHub
2. Click the **"Actions"** tab
3. Select the latest workflow run
4. View results for each OS/Python combination

Each job runs:
- All 31 test cases from `grid_counting_tests.py`
- Hardware detection (confirms CPU-only environment)
- Performance validation

#### CI Test Environment

- **Hardware**: CPU-only runners (no GPUs available)
- **Libraries**: NumPy, PyTorch (CPU), JAX (CPU)
- **Purpose**: Validates cross-platform compatibility and correctness

### Expected Test Results

#### Successful Run

```
Running tests for 2D array positive value counting...
--------------------------------------------------
[31 test cases run]
--------------------------------------------------
✓ All tests passed!
```

### Troubleshooting

#### Tests Fail Locally

1. **Missing dependencies**: Ensure NumPy is installed (`pip install numpy`)
2. **Python version**: Requires Python 3.9+ (tested on 3.9, 3.10, 3.11)
3. **Import errors**: Check that `grid_counting.py` is in the same directory

#### CI Tests Fail

1. Check GitHub Actions logs for specific error
2. Verify code works locally first
3. Check for OS-specific issues (path separators, line endings)
4. Ensure all dependencies are in `requirements.txt`

#### Browser Tests Don't Run

1. Hard refresh browser (Ctrl+Shift+R or Cmd+Shift+R)
2. Check browser console for errors
3. Ensure JavaScript is enabled
4. Try different browser if issues persist

### Test File Locations

- **Main test suite**: `grid_counting_tests.py`
- **Browser test runner**: `docs/grid_counting_tests.py` (same tests, browser-optimized)
- **Test runner JavaScript**: `docs/test_runner.js`
- **CI workflow**: `.github/workflows/test.yml`

## License

This code is provided as-is for demonstration purposes.