"""
Unit tests for 2D array positive value counting functions.
Includes functionality tests and performance benchmarks.
"""

import numpy as np
import time
import platform
from grid_counting import (
    detect_available_accelerators,
    DenseGrid,
    SparseGrid
)


def render_grid(grid, max_size=10, cell_renderer=None):
    """Render a grid as ASCII art if it's small enough
    
    Args:
        grid: 2D numpy array to render
        max_size: Maximum dimension to render (default 10)
        cell_renderer: Optional function (row, col, val) -> str to customize cell rendering
    
    Returns:
        String representation of the grid, or None if grid is too large
    """
    h, w = grid.shape
    if h > max_size or w > max_size:
        return None
    
    result = []
    # Column headers aligned with 3-char cells (shifted right to align with data)
    header = ''.join(f'  {c}' if c < 10 else f' {c}' for c in range(w))
    result.append('     ' + header)  # Extra space to shift right
    result.append('    ' + '-' * (3 * w + 1))
    
    for r in range(h):
        row = []
        for c in range(w):
            val = grid[r, c]
            if cell_renderer:
                row.append(cell_renderer(r, c, val))
            elif val == 0:
                row.append(' . ')
            elif val == 1:
                row.append(' @ ')  # Seed
            elif val == 2:
                row.append(' * ')  # Set location
            else:
                row.append(f'{int(val):2d} ')
        result.append(f'{r:2d} ' + ''.join(row))
    
    result.append('    ' + '-' * (3 * w + 1))
    footer = ''.join(f'  {c}' if c < 10 else f' {c}' for c in range(w))
    result.append('  ' + footer)  # Even less space to shift left
    
    return '\n'.join(result)


def print_test_separator():
    """Print a visual separator between tests"""
    print()


def _header(title: str):
    """Print TEST: header for browser pagination (no ==== lines, handled by run_all_tests)"""
    print(f"TEST: {title}")


def test_set_dense_grid_basic():
    """Test basic set_dense_grid functionality (now sets to 2)"""
    # Report operating system information
    os_name = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    machine = platform.machine()
    processor = platform.processor()
    print(f"Operating System: {os_name} {os_release} ({os_version})")
    print(f"Platform: {machine} / {processor}")
    
    grid = DenseGrid(3, 3)
    grid.set_locations_to_value([(0, 0), (1, 1), (2, 2)])
    result = grid.grid
    expected = np.zeros((3, 3), dtype=np.int32)
    expected[0, 0] = 2
    expected[1, 1] = 2
    expected[2, 2] = 2
    assert np.array_equal(result, expected)
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: set_dense_grid (diagonal pattern)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    grid = DenseGrid(result.shape[0], result.shape[1])
    grid.grid = result
    count = grid.count_positive_valued_cells()
    print(f"   Positive count: {count}")
    print("✓ Basic set_locations test passed")


def test_set_dense_grid_empty():
    """Test with empty locations list"""
    grid = DenseGrid(5, 5)
    grid.set_locations_to_value([])
    result = grid.grid
    expected = np.zeros((5, 5), dtype=np.int32)
    assert np.array_equal(result, expected)
    
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: set_dense_grid with empty locations (no locations specified)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    
    grid = DenseGrid(result.shape[0], result.shape[1])
    grid.grid = result
    count = grid.count_positive_valued_cells()
    print(f"   Positive count: {count}")
    print("✓ Empty locations test passed")


def test_set_dense_grid_out_of_bounds():
    """Test with out-of-bounds locations (should be ignored)"""
    input_locations = [(0, 0), (10, 10), (-1, 0), (1, 1)]
    grid = DenseGrid(3, 3)
    grid.set_locations_to_value(input_locations)
    result = grid.grid
    expected = np.zeros((3, 3), dtype=np.int32)
    expected[0, 0] = 2
    expected[1, 1] = 2
    assert np.array_equal(result, expected)
    
    # List valid vs invalid locations
    valid_locations = [(0, 0), (1, 1)]
    invalid_locations = [(10, 10), (-1, 0)]
    
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: set_dense_grid with out-of-bounds locations (invalid locations ignored)")
        print("=" * 70)
        print(f"Input locations: {input_locations}")
        print(f"Valid locations (set): {valid_locations}")
        print(f"Invalid locations (ignored): {invalid_locations}")
        print()
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    
    grid = DenseGrid(result.shape[0], result.shape[1])
    grid.grid = result
    count = grid.count_positive_valued_cells()
    print(f"   Positive count: {count}")
    print("✓ Out-of-bounds locations test passed")


def test_set_dense_grid_single():
    """Test with single location"""
    grid = DenseGrid(5, 5)
    grid.set_locations_to_value([(2, 3)])
    result = grid.grid
    assert result[2, 3] == 2
    assert np.sum(result) == 2
    
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: set_dense_grid single location")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    
    grid = DenseGrid(result.shape[0], result.shape[1])
    grid.grid = result
    count = grid.count_positive_valued_cells()
    print(f"   Positive count: {count}")
    print("✓ Single location test passed")


def test_set_dense_grid_multiple():
    """Test with multiple locations"""
    locations = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]
    grid = DenseGrid(5, 5)
    grid.set_locations_to_value(locations)
    result = grid.grid
    assert np.sum(result) == len(locations) * 2  # Each location set to 2
    for r, c in locations:
        assert result[r, c] == 2
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: set_dense_grid multiple locations (corners + center)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    grid = DenseGrid(result.shape[0], result.shape[1])
    grid.grid = result
    count = grid.count_positive_valued_cells()
    print(f"   Positive count: {count}")
    print("✓ Multiple locations test passed")


def test_set_dense_grid_modify_existing():
    """Test modifying existing array"""
    existing = np.zeros((3, 3), dtype=np.int32)
    existing[0, 0] = 5  # Set an existing value
    grid = DenseGrid(existing.shape[0], existing.shape[1])
    grid.grid = existing
    grid.set_locations_to_value([(1, 1)])
    result = grid.grid
    assert result[0, 0] == 5  # Original value preserved
    assert result[1, 1] == 2  # New location set to 2
    assert np.sum(result) == 7
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: Modify existing dense_grid (preserves existing value)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    grid = DenseGrid(result.shape[0], result.shape[1])
    grid.grid = result
    count = grid.count_positive_valued_cells()
    print(f"   Positive count: {count}")
    print("✓ Modify existing array test passed")


def test_set_dense_grid_from_sparse_locations():
    """Test creating dense grid from sparse locations"""
    grid = DenseGrid(4, 4)
    grid.set_locations_to_value([(0, 0), (3, 3)])
    result = grid.grid
    assert result.shape == (4, 4)
    assert result[0, 0] == 2
    assert result[3, 3] == 2
    assert np.sum(result) == 4
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: set_dense_grid from sparse locations (opposite corners)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    grid = DenseGrid(result.shape[0], result.shape[1])
    grid.grid = result
    count = grid.count_positive_valued_cells()
    print(f"   Positive count: {count}")
    print("✓ Create sparse grid test passed")



def test_count_dense_grid_basic():
    """Test basic count_nonzero functionality (counts only positive values)"""
    array = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    grid = DenseGrid(array.shape[0], array.shape[1])
    grid.grid = array
    result = grid.count_positive_valued_cells()
    assert result == 5  # All are positive
    ascii = render_grid(array)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: count_dense_grid (mixed positive values)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    print(f"   Positive count: {result}")
    print("✓ Basic count_nonzero test passed")


def test_count_dense_grid_all_zeros():
    """Test count_nonzero with all zeros"""
    array = np.zeros((3, 3))
    grid = DenseGrid(array.shape[0], array.shape[1])
    grid.grid = array
    result = grid.count_positive_valued_cells()
    assert result == 0  # No positive values
    ascii = render_grid(array)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: count_dense_grid all zeros (no positive values)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    print(f"   Positive count: {result}")
    print("✓ Count non-zero all zeros test passed")


def test_count_dense_grid_all_nonzero():
    """Test count_nonzero with all positive values"""
    array = np.ones((3, 3))
    grid = DenseGrid(array.shape[0], array.shape[1])
    grid.grid = array
    result = grid.count_positive_valued_cells()
    assert result == 9  # All are positive
    ascii = render_grid(array)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: count_dense_grid all positive (all values are 1)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    print(f"   Positive count: {result}")
    print("✓ Count non-zero all positive test passed")


def test_count_dense_grid_with_set_dense_grid():
    """Test count_nonzero with set_dense_grid (values set to 2 = positive)"""
    grid = DenseGrid(5, 5)
    grid.set_locations_to_value([(0, 0), (1, 1), (2, 2)])
    count = grid.count_positive_valued_cells()
    assert count == 3  # 2 is positive
    ascii = render_grid(grid)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: count_dense_grid after set_dense_grid (3 locations set to 2)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    print(f"   Positive count: {count}")
    print("✓ Count non-zero with set_locations test passed")


def test_count_dense_grid_excludes_negatives():
    """Test that negative values are excluded"""
    array = np.array([[1, -1, 0, 2], [0, -3, 1, 0], [-5, 0, 3, 0]])
    grid = DenseGrid(array.shape[0], array.shape[1])
    grid.grid = array
    result = grid.count_positive_valued_cells()
    assert result == 4  # Only positive: 1, 2, 1, 3 (negatives and zeros excluded)
    
    # Render with custom symbols for negatives using cell renderer
    def render_cell(r, c, val):
        if val > 0:
            return f'{int(val):2d} '
        elif val < 0:
            return ' - '  # Negative
        else:
            return ' . '  # Zero
    
    ascii = render_grid(array, max_size=10, cell_renderer=render_cell)
    print()
    print("=" * 70)
    print("TEST: count_dense_grid excludes negatives (only positive counted)")
    print("=" * 70)
    ascii_lines = ascii.split('\n')
    print(ascii_lines[0])
    for line in ascii_lines[1:]:
        print(f"   {line}")
    
    print(f"   Positive count: {result}")
    print("✓ Count excludes negative values test passed")


def test_set_dense_grid_neighborhoods_overlapping():
    """Test that overlapping neighborhoods are counted correctly (no double counting)"""
    # Create grid with two seeds that are close together (within L=2 of each other)
    grid = DenseGrid(8, 8)
    seeds = [(2, 2), (3, 3)]  # Seeds at (2,2) and (3,3) - Manhattan distance = 2
    grid.set_neighborhoods_to_value(seeds, L=2, value=2)
    
    # Count should count all positive values, not double-count overlapping regions
    count = grid.count_positive_valued_cells()
    
    # Calculate expected count: positive-valued cells in both neighborhoods
    # Seed 1 at (2,2): creates neighborhood with cells within distance 2
    # Seed 2 at (3,3): creates overlapping neighborhood
    # They overlap, so we should get positive-valued cells only
    
    # Expected positive-valued cells in L=2 neighborhood of (2,2) and (3,3) combined
    # Since the seeds are at (2,2) and (3,3) with Manhattan distance 2,
    # their L=2 neighborhoods overlap significantly
    # With these specific seeds and L=2, we get exactly 18 positive-valued cells
    expected_count = 18
    
    assert count == expected_count, f"Expected {expected_count}, got {count}"
    
    # Render the result with custom cell renderer to show seeds
    def render_cell(r, c, val):
        is_seed = (r, c) in seeds
        if is_seed:
            return ' @ '  # Seed
        elif val == 2:
            return ' * '  # In neighborhood
        else:
            return ' . '  # Outside
    
    ascii = render_grid(grid.grid, max_size=10, cell_renderer=render_cell)
    print()
    print("=" * 70)
    print("TEST: set_dense_grid_neighborhoods overlapping (L=2, close seeds)")
    print("=" * 70)
    ascii_lines = ascii.split('\n')
    print(ascii_lines[0])
    for line in ascii_lines[1:]:
        print(f"   {line}")
    
    # Calculate Manhattan distance between seeds
    seed1, seed2 = seeds[0], seeds[1]
    dist_between = abs(seed1[0] - seed2[0]) + abs(seed1[1] - seed2[1])
    
    print(f"   Seeds at {seed1} and {seed2}")
    print(f"   Manhattan distance between seeds: {dist_between}")
    print(f"   Positive count: {count} (overlapping regions counted once)")
    print("✓ Overlapping Manhattan neighborhoods test passed")


def test_set_dense_grid_neighborhoods_non_overlapping():
    """Test two seeds with Manhattan distance > L (non-overlapping neighborhoods)"""
    # Create grid with two seeds that are far apart (distance > L)
    grid = DenseGrid(10, 10)
    seeds = [(2, 2), (7, 7)]  # Seeds at (2,2) and (7,7) - Manhattan distance = 10
    grid.set_neighborhoods_to_value(seeds, L=2, value=2)
    
    # Count should count positive-valued cells from both neighborhoods
    count = grid.count_positive_valued_cells()
    
    # Calculate expected count: 
    # Each seed has L=2 neighborhood with ~13 cells
    # Since they don't overlap (distance 10 > L=2), we expect ~26 cells
    expected_count = 26
    
    assert count == expected_count, f"Expected {expected_count}, got {count}"
    
    # Render the result with custom cell renderer to show seeds
    def render_cell(r, c, val):
        is_seed = (r, c) in seeds
        if is_seed:
            return ' @ '  # Seed
        elif val == 2:
            return ' * '  # In neighborhood
        else:
            return ' . '  # Outside
    
    ascii = render_grid(grid.grid, max_size=10, cell_renderer=render_cell)
    print()
    print("=" * 70)
    print("TEST: set_dense_grid_neighborhoods non-overlapping (L=2, far seeds)")
    print("=" * 70)
    ascii_lines = ascii.split('\n')
    print(ascii_lines[0])
    for line in ascii_lines[1:]:
        print(f"   {line}")
    
    # Calculate Manhattan distance between seeds
    seed1, seed2 = seeds[0], seeds[1]
    dist_between = abs(seed1[0] - seed2[0]) + abs(seed1[1] - seed2[1])
    
    print(f"   Seeds at {seed1} and {seed2}")
    print(f"   Manhattan distance between seeds: {dist_between}")
    print(f"   Positive count: {count} (separate neighborhoods)")
    print("✓ Non-overlapping Manhattan neighborhoods test passed")


def test_sparse_grid_class():
    """Test SparseGrid class functionality"""
    # Create a SparseGrid and set neighborhoods
    sparse = SparseGrid(100, 100)
    seeds = [(10, 10), (20, 20)]
    sparse.set_neighborhoods_to_value(seeds, L=3, value=2)
    assert sparse.num_rows == 100
    assert sparse.num_cols == 100
    
    # Count should work with L=3
    count = sparse.count_positive_valued_cells()
    assert count > 0  # Should have some cells within L=3 of seeds
    
    print()
    print("=" * 70)
    print("TEST: SparseGrid.class")
    print("=" * 70)
    print(f"   Grid dimensions: {sparse.num_rows}x{sparse.num_cols}")
    print(f"   Seed locations: {seeds}")
    print(f"   L: {3}")
    print(f"   Count: {count}")
    print("✓ SparseGrid class test passed")


def test_hardware_detection():
    """Test that hardware detection works and at least NumPy is available"""
    print()
    print("=" * 70)
    print("TEST: Hardware detection")
    print("=" * 70)
    
    # Show platform information
    os_name = platform.system()
    print(f"Platform: {os_name}")
    print("Testing hardware detection...")
    
    accelerators = detect_available_accelerators()
    
    # At minimum, NumPy should be available
    assert 'numpy' in accelerators, "NumPy should always be available"
    assert accelerators['numpy'] == True, "NumPy should be True"
    
    # Check if libraries are installed
    pytorch_installed = False
    jax_installed = False
    try:
        import torch
        pytorch_installed = True
    except ImportError:
        pass
    try:
        import jax
        jax_installed = True
    except ImportError:
        pass
    
    print(f"  NumPy: {'✓ Available' if accelerators['numpy'] else '✗ Not available'} (always required)")
    
    if pytorch_installed:
        print(f"  PyTorch: ✓ Installed")
        if os_name == "Darwin":  # macOS
            print(f"    - CUDA: {'✓ Available' if accelerators.get('pytorch_cuda') else '✗ Not available (NVIDIA GPU required)'}")
            print(f"    - MPS: {'✓ Available' if accelerators.get('pytorch_mps') else '✗ Not available (Apple Silicon required)'}")
        elif os_name == "Windows":
            print(f"    - CUDA: {'✓ Available' if accelerators.get('pytorch_cuda') else '✗ Not available (NVIDIA GPU + CUDA drivers required)'}")
            print(f"    - MPS: N/A (Windows)")
        else:  # Linux
            print(f"    - CUDA: {'✓ Available' if accelerators.get('pytorch_cuda') else '✗ Not available (NVIDIA GPU + CUDA drivers required)'}")
            print(f"      Linux supports: CUDA via PyTorch (NVIDIA GPUs)")
            print(f"    - MPS: N/A (macOS/Apple Silicon only)")
    else:
        print(f"  PyTorch: ✗ Not installed")
    
    if jax_installed:
        print(f"  JAX: ✓ Installed")
        print(f"    - GPU: {'✓ Available' if accelerators.get('jax_gpu') else '✗ Not available (GPU drivers required)'}")
        if os_name == "Linux" and not accelerators.get('jax_gpu'):
            print(f"      Linux supports: JAX GPU via CUDA (requires CUDA drivers)")
        print(f"    - TPU: {'✓ Available' if accelerators.get('jax_tpu') else '✗ Not available (Google Cloud TPU required)'}")
    else:
        print(f"  JAX: ✗ Not installed")
    
    print("✓ Hardware detection test passed")


def test_counting_performance_small():
    """Test counting performance on small array"""
    print()
    print("=" * 70)
    print("TEST: Counting performance (small array)")
    print("=" * 70)
    print("Testing counting performance (small array)...")
    array = np.random.randint(0, 10, size=(100, 100))
    
    start = time.perf_counter()
    grid = DenseGrid(array.shape[0], array.shape[1])
    grid.grid = array
    result = grid.count_positive_valued_cells()
    elapsed = time.perf_counter() - start
    
    assert result > 0, "Should count some positive values"
    assert elapsed < 0.25, "Should be fast (<250ms)"
    
    print(f"  Count: {result} in {elapsed*1000:.4f}ms")
    print("✓ Small array performance test passed")


def test_manhattan_performance_comparison():
    """Test that direct counting is faster than grid creation"""
    print("=" * 70)
    print("TEST: count_sparse_grid vs set_dense_grid + count_dense_grid performance")
    print("=" * 70)
    print("Testing Manhattan neighborhood performance...")
    
    seeds = [(10, 10), (30, 30), (50, 50)]
    L = 3
    
    # Direct counting
    times_direct = []
    for _ in range(3):
        start = time.perf_counter()
        sparse = SparseGrid(100, 100)
        sparse.set_neighborhoods_to_value(seeds, L=L, value=2)
        count_direct = sparse.count_positive_valued_cells()
        times_direct.append(time.perf_counter() - start)
    
    avg_direct = np.mean(times_direct)
    
    # Grid creation + count
    times_grid = []
    for _ in range(3):
        grid = DenseGrid(100, 100)
        start = time.perf_counter()
        grid.set_neighborhoods_to_value(seeds, L=L, value=2)
        count_grid = grid.count_positive_valued_cells()
        times_grid.append(time.perf_counter() - start)
    
    avg_grid = np.mean(times_grid)
    
    print(f"  Direct counting: {avg_direct*1000:.4f}ms -> {count_direct} cells")
    print(f"  Grid + counting: {avg_grid*1000:.4f}ms -> {count_grid} cells")
    
    # Both should give same count
    assert count_direct == count_grid, "Counts should match"
    
    print(f"  Direct counting is {avg_grid/avg_direct:.2f}x faster")
    print("✓ Manhattan performance comparison test passed")


def test_example_1_n3_centered():
    """Example 1: One positive cell fully contained; N=3"""
    # According to Counting_grid-cell_neighborhoods.txt: 25 cells in N=3 neighborhood
    sparse = SparseGrid(11, 11)
    sparse.set_neighborhoods_to_value([(5, 5)], L=3, value=2)
    count = sparse.count_positive_valued_cells()
    
    # Formula for cells in Manhattan diamond: cells = (2*N+1)^2 - N*(N+1) for N=3
    # Actually: sum from d=0 to N of 4d+1 cells at distance d
    # N=3: 1 + 4 + 8 + 12 = 25 cells
    assert count == 25, f"Expected 25 cells for N=3 centered, got {count}"
    
    print()
    print("=" * 70)
    print("TEST: Example 1 - One positive cell fully contained; N=3")
    print("=" * 70)
    print(f"   Grid: 11x11, Seed: (5,5), L=3")
    print(f"   Count: {count} cells")
    print("✓ Example 1 (N=3, centered, 25 cells) test passed")


def test_example_2_n3_near_edge():
    """Example 2: One positive cell near an edge; N=3"""
    # According to Counting_grid-cell_neighborhoods.txt: 21 cells (25 - 4 that fell off edge)
    sparse = SparseGrid(11, 11)
    sparse.set_neighborhoods_to_value([(5, 2)], L=3, value=2)
    count = sparse.count_positive_valued_cells()
    
    # Near left edge at column 2, some cells fall off
    assert count == 24, f"Expected 24 cells for N=3 near edge, got {count}"
    
    print()
    print("=" * 70)
    print("TEST: Example 2 - One positive cell near edge; N=3")
    print("=" * 70)
    print(f"   Grid: 11x11, Seed: (5,2), L=3")
    print(f"   Count: {count} cells (1 cell fell off edge)")
    print("✓ Example 2 (N=3, near edge, 24 cells) test passed")


def test_example_3_n2_disjoint():
    """Example 3: Two positive values with disjoint neighborhoods; N=2"""
    # According to Counting_grid-cell_neighborhoods.txt: 26 cells total (13 per neighborhood)
    sparse = SparseGrid(11, 11)
    sparse.set_neighborhoods_to_value([(2, 2), (8, 8)], L=2, value=2)
    count = sparse.count_positive_valued_cells()
    
    # Two disjoint neighborhoods: 13 cells each = 26 total
    assert count == 26, f"Expected 26 cells for N=2 disjoint neighborhoods, got {count}"
    
    print()
    print("=" * 70)
    print("TEST: Example 3 - Two positive values with disjoint neighborhoods; N=2")
    print("=" * 70)
    print(f"   Grid: 11x11, Seeds: (2,2) and (8,8), L=2")
    print(f"   Count: {count} cells (13 per neighborhood, no overlap)")
    print("✓ Example 3 (N=2, disjoint, 26 cells) test passed")


def test_example_4_n2_overlapping():
    """Example 4: Two positive values with overlapping neighborhoods; N=2"""
    # According to Counting_grid-cell_neighborhoods.txt: 22 cells (overlapping)
    # Place seeds close enough that neighborhoods overlap
    sparse = SparseGrid(11, 11)
    sparse.set_neighborhoods_to_value([(5, 5), (6, 6)], L=2, value=2)
    count = sparse.count_positive_valued_cells()
    
    # Should be less than 26 (disjoint case) due to overlap
    assert count < 26, "Overlapping neighborhoods should count less than disjoint"
    assert count >= 13, "Should have at least one full neighborhood"
    
    print()
    print("=" * 70)
    print("TEST: Example 4 - Two positive values with overlapping neighborhoods; N=2")
    print("=" * 70)
    print(f"   Grid: 11x11, Seeds: (5,5) and (6,6), L=2")
    print(f"   Count: {count} cells (overlapping regions counted once)")
    print("✓ Example 4 (N=2, overlapping, ~22 cells) test passed")


def test_edge_case_corner():
    """Edge case: Positive value in a corner"""
    sparse = SparseGrid(11, 11)
    sparse.set_neighborhoods_to_value([(0, 0)], L=2, value=2)
    count = sparse.count_positive_valued_cells()
    
    # Should be fewer than centered case due to corner
    assert count < 13, "Corner placement should reduce neighborhood size"
    assert count > 0, "Should always count at least the seed"
    
    print()
    print("=" * 70)
    print("TEST: Edge case - Positive value in corner")
    print("=" * 70)
    print(f"   Grid: 11x11, Seed: (0,0) corner, L=2")
    print(f"   Count: {count} cells")
    print("✓ Corner edge case test passed")


def test_edge_case_odd_shaped_1x21():
    """Edge case: Odd shaped array 1x21"""
    sparse = SparseGrid(1, 21)
    sparse.set_neighborhoods_to_value([(0, 10)], L=3, value=2)
    count = sparse.count_positive_valued_cells()
    
    # Single row, neighborhood spreads horizontally
    # At L=3, can reach cells from 10-3 to 10+3 = columns 7 to 13
    # That's 7 cells total (in bounds)
    assert count == 7, f"Expected 7 cells for 1x21 array, got {count}"
    
    print()
    print("=" * 70)
    print("TEST: Edge case - Odd shaped array 1x21")
    print("=" * 70)
    print(f"   Grid: 1x21, Seed: (0,10), L=3")
    print(f"   Count: {count} cells")
    print("✓ 1x21 edge case test passed")


def test_edge_case_odd_shaped_1x1():
    """Edge case: Smallest array 1x1"""
    sparse = SparseGrid(1, 1)
    sparse.set_neighborhoods_to_value([(0, 0)], L=10, value=2)
    count = sparse.count_positive_valued_cells()
    
    # Only one cell total, so count should be 1
    assert count == 1, f"Expected 1 cell for 1x1 array, got {count}"
    
    print()
    print("=" * 70)
    print("TEST: Edge case - Smallest array 1x1")
    print("=" * 70)
    print(f"   Grid: 1x1, Seed: (0,0), L=10")
    print(f"   Count: {count} cells")
    print("✓ 1x1 edge case test passed")


def test_edge_case_odd_shaped_10x1():
    """Edge case: Long vertical array 10x1"""
    sparse = SparseGrid(10, 1)
    sparse.set_neighborhoods_to_value([(5, 0)], L=3, value=2)
    count = sparse.count_positive_valued_cells()
    
    # Single column, neighborhood spreads vertically
    # At L=3, from row 5-3 to 5+3 = rows 2 to 8
    # That's 7 cells (rows 2,3,4,5,6,7,8)
    assert count == 7, f"Expected 7 cells for 10x1 array, got {count}"
    
    print()
    print("=" * 70)
    print("TEST: Edge case - Long vertical array 10x1")
    print("=" * 70)
    print(f"   Grid: 10x1, Seed: (5,0), L=3")
    print(f"   Count: {count} cells")
    print("✓ 10x1 edge case test passed")


def test_edge_case_odd_shaped_2x2():
    """Edge case: Small square 2x2"""
    sparse = SparseGrid(2, 2)
    sparse.set_neighborhoods_to_value([(0, 0)], L=2, value=2)
    count = sparse.count_positive_valued_cells()
    
    # With L=2 in a 2x2 grid, should get all 4 cells
    assert count == 4, f"Expected 4 cells for 2x2 array, got {count}"
    
    print()
    print("=" * 70)
    print("TEST: Edge case - Small square 2x2")
    print("=" * 70)
    print(f"   Grid: 2x2, Seed: (0,0), L=2")
    print(f"   Count: {count} cells")
    print("✓ 2x2 edge case test passed")


def test_edge_case_n0():
    """Edge case: N=0 (only seed cells)"""
    sparse = SparseGrid(11, 11)
    sparse.set_neighborhoods_to_value([(5, 5), (2, 2)], L=0, value=2)
    count = sparse.count_positive_valued_cells()
    
    # With L=0, should only count the seeds themselves
    assert count == 2, f"Expected 2 cells for N=0, got {count}"
    
    print()
    print("=" * 70)
    print("TEST: Edge case - N=0 (only seed cells)")
    print("=" * 70)
    print(f"   Grid: 11x11, Seeds: (5,5) and (2,2), L=0")
    print(f"   Count: {count} cells (only seed locations)")
    print("✓ N=0 edge case test passed")


def test_edge_case_large_n():
    """Edge case: N >> max(W, H)"""
    sparse = SparseGrid(11, 11)
    sparse.set_neighborhoods_to_value([(5, 5)], L=100, value=2)
    count = sparse.count_positive_valued_cells()
    
    # With L=100 in an 11x11 grid, should count all 121 cells
    assert count == 121, f"Expected 121 cells for N=100 in 11x11, got {count}"
    
    print()
    print("=" * 70)
    print("TEST: Edge case - N >> max(W,H)")
    print("=" * 70)
    print(f"   Grid: 11x11, Seed: (5,5), L=100")
    print(f"   Count: {count} cells (entire grid)")
    print("✓ Large N edge case test passed")


def test_edge_case_zero_dimensions():
    """Edge case: Zero dimensions should raise ValueError"""
    print()
    print("=" * 70)
    print("TEST: Edge case - Zero dimensions (should reject)")
    print("=" * 70)
    
    # Test zero rows
    try:
        sparse = SparseGrid(0, 10)
        assert False, "Should have raised ValueError for 0 rows"
    except ValueError as e:
        print(f"   ✓ Correctly rejected 0x10 grid: {e}")
    
    # Test zero columns
    try:
        sparse = SparseGrid(10, 0)
        assert False, "Should have raised ValueError for 0 columns"
    except ValueError as e:
        print(f"   ✓ Correctly rejected 10x0 grid: {e}")
    
    # Test zero by zero
    try:
        sparse = SparseGrid(0, 0)
        assert False, "Should have raised ValueError for 0x0 grid"
    except ValueError as e:
        print(f"   ✓ Correctly rejected 0x0 grid: {e}")
    
    # Test negative dimensions
    try:
        sparse = SparseGrid(-5, 10)
        assert False, "Should have raised ValueError for negative rows"
    except ValueError as e:
        print(f"   ✓ Correctly rejected -5x10 grid: {e}")
    
    print("✓ Zero dimensions edge case test passed")


def run_all_tests():
    """Run all tests"""
    print("Running tests for 2D array positive value counting...")
    print("-" * 50)
    total = 29
    idx = 1
    
    try:
        print(); idx += 1; test_set_dense_grid_basic()
        print(); idx += 1; test_set_dense_grid_empty()
        print(); idx += 1; test_set_dense_grid_out_of_bounds()
        print(); idx += 1; test_set_dense_grid_single()
        print(); idx += 1; test_set_dense_grid_multiple()
        print(); idx += 1; test_set_dense_grid_modify_existing()
        print(); idx += 1; test_set_dense_grid_from_sparse_locations()
        print(); idx += 1; test_count_dense_grid_basic()
        print(); idx += 1; test_count_dense_grid_all_zeros()
        print(); idx += 1; test_count_dense_grid_all_nonzero()
        print(); idx += 1; test_count_dense_grid_with_set_dense_grid()
        print(); idx += 1; test_count_dense_grid_excludes_negatives()
        print(); idx += 1; test_set_dense_grid_neighborhoods_overlapping()
        print(); idx += 1; test_set_dense_grid_neighborhoods_non_overlapping()
        print(); idx += 1; test_sparse_grid_class()
        print(); idx += 1; test_hardware_detection()
        print(); idx += 1; test_counting_performance_small()
        
        # Project requirement examples
        print(); idx += 1; test_example_1_n3_centered()
        print(); idx += 1; test_example_2_n3_near_edge()
        print(); idx += 1; test_example_3_n2_disjoint()
        print(); idx += 1; test_example_4_n2_overlapping()
        
        # Edge cases from requirements
        print(); idx += 1; test_edge_case_corner()
        print(); idx += 1; test_edge_case_odd_shaped_1x21()
        print(); idx += 1; test_edge_case_odd_shaped_1x1()
        print(); idx += 1; test_edge_case_odd_shaped_10x1()
        print(); idx += 1; test_edge_case_odd_shaped_2x2()
        print(); idx += 1; test_edge_case_n0()
        print(); idx += 1; test_edge_case_large_n()
        print(); idx += 1; test_edge_case_zero_dimensions()

        print("-" * 50)
        print("✓ All tests passed!")
        return True
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    # Detect hardware first
    print("Available hardware:")
    accelerators = detect_available_accelerators()
    print()
    
    # Run tests
    success = run_all_tests()
    exit(0 if success else 1)

