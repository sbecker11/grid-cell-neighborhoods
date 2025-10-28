"""
Unit tests for 2D array positive value counting functions.
Includes functionality tests and performance benchmarks.
"""

import numpy as np
import time
from grid_counting import (
    detect_available_accelerators,
    FullGrid,
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


def test_set_full_grid_basic():
    """Test basic set_full_grid functionality (now sets to 2)"""
    grid = FullGrid((3, 3))
    grid.set_locations([(0, 0), (1, 1), (2, 2)])
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
        print("TEST: set_full_grid (diagonal pattern)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    grid = FullGrid(result)
    count = grid.count_positive_values()
    print(f"   Positive count: {count}")
    print("✓ Basic set_locations test passed")


def test_set_full_grid_empty():
    """Test with empty locations list"""
    grid = FullGrid((5, 5))
    grid.set_locations([])
    result = grid.grid
    expected = np.zeros((5, 5), dtype=np.int32)
    assert np.array_equal(result, expected)
    
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: set_full_grid with empty locations (no locations specified)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    
    grid = FullGrid(result)
    count = grid.count_positive_values()
    print(f"   Positive count: {count}")
    print("✓ Empty locations test passed")


def test_set_full_grid_out_of_bounds():
    """Test with out-of-bounds locations (should be ignored)"""
    input_locations = [(0, 0), (10, 10), (-1, 0), (1, 1)]
    grid = FullGrid((3, 3))
    grid.set_locations(input_locations)
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
        print("TEST: set_full_grid with out-of-bounds locations (invalid locations ignored)")
        print("=" * 70)
        print(f"Input locations: {input_locations}")
        print(f"Valid locations (set): {valid_locations}")
        print(f"Invalid locations (ignored): {invalid_locations}")
        print()
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    
    grid = FullGrid(result)
    count = grid.count_positive_values()
    print(f"   Positive count: {count}")
    print("✓ Out-of-bounds locations test passed")


def test_set_full_grid_single():
    """Test with single location"""
    grid = FullGrid((5, 5))
    grid.set_locations([(2, 3)])
    result = grid.grid
    assert result[2, 3] == 2
    assert np.sum(result) == 2
    
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: set_full_grid single location")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    
    grid = FullGrid(result)
    count = grid.count_positive_values()
    print(f"   Positive count: {count}")
    print("✓ Single location test passed")


def test_set_full_grid_multiple():
    """Test with multiple locations"""
    locations = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]
    grid = FullGrid((5, 5))
    grid.set_locations(locations)
    result = grid.grid
    assert np.sum(result) == len(locations) * 2  # Each location set to 2
    for r, c in locations:
        assert result[r, c] == 2
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: set_full_grid multiple locations (corners + center)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    grid = FullGrid(result)
    count = grid.count_positive_values()
    print(f"   Positive count: {count}")
    print("✓ Multiple locations test passed")


def test_set_full_grid_modify_existing():
    """Test modifying existing array"""
    existing = np.zeros((3, 3), dtype=np.int32)
    existing[0, 0] = 5  # Set an existing value
    grid = FullGrid(existing)
    grid.set_locations([(1, 1)])
    result = grid.grid
    assert result[0, 0] == 5  # Original value preserved
    assert result[1, 1] == 2  # New location set to 2
    assert np.sum(result) == 7
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: Modify existing full_grid (preserves existing value)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    grid = FullGrid(result)
    count = grid.count_positive_values()
    print(f"   Positive count: {count}")
    print("✓ Modify existing array test passed")


def test_set_full_grid_from_sparse_locations():
    """Test creating full grid from sparse locations"""
    grid = FullGrid((4, 4))
    grid.set_locations([(0, 0), (3, 3)])
    result = grid.grid
    assert result.shape == (4, 4)
    assert result[0, 0] == 2
    assert result[3, 3] == 2
    assert np.sum(result) == 4
    ascii = render_grid(result)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: set_full_grid from sparse locations (opposite corners)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    grid = FullGrid(result)
    count = grid.count_positive_values()
    print(f"   Positive count: {count}")
    print("✓ Create sparse grid test passed")



def test_count_full_grid_basic():
    """Test basic count_nonzero functionality (counts only positive values)"""
    array = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    grid = FullGrid(array)
    result = grid.count_positive_values()
    assert result == 5  # All are positive
    ascii = render_grid(array)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: count_full_grid (mixed positive values)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    print(f"   Positive count: {result}")
    print("✓ Basic count_nonzero test passed")


def test_count_full_grid_all_zeros():
    """Test count_nonzero with all zeros"""
    array = np.zeros((3, 3))
    grid = FullGrid(array)
    result = grid.count_positive_values()
    assert result == 0  # No positive values
    ascii = render_grid(array)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: count_full_grid all zeros (no positive values)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    print(f"   Positive count: {result}")
    print("✓ Count non-zero all zeros test passed")


def test_count_full_grid_all_nonzero():
    """Test count_nonzero with all positive values"""
    array = np.ones((3, 3))
    grid = FullGrid(array)
    result = grid.count_positive_values()
    assert result == 9  # All are positive
    ascii = render_grid(array)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: count_full_grid all positive (all values are 1)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    print(f"   Positive count: {result}")
    print("✓ Count non-zero all positive test passed")


def test_count_full_grid_with_set_full_grid():
    """Test count_nonzero with set_full_grid (values set to 2 = positive)"""
    grid = FullGrid((5, 5))
    grid.set_locations([(0, 0), (1, 1), (2, 2)])
    count = grid.count_positive_values()
    assert count == 3  # 2 is positive
    ascii = render_grid(grid)
    if ascii:
        print()
        print("=" * 70)
        print("TEST: count_full_grid after set_full_grid (3 locations set to 2)")
        print("=" * 70)
        ascii_lines = ascii.split('\n')
        print(ascii_lines[0])
        for line in ascii_lines[1:]:
            print(f"   {line}")
    print(f"   Positive count: {count}")
    print("✓ Count non-zero with set_locations test passed")


def test_count_full_grid_excludes_negatives():
    """Test that negative values are excluded"""
    array = np.array([[1, -1, 0, 2], [0, -3, 1, 0], [-5, 0, 3, 0]])
    grid = FullGrid(array)
    result = grid.count_positive_values()
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
    print("TEST: count_full_grid excludes negatives (only positive counted)")
    print("=" * 70)
    ascii_lines = ascii.split('\n')
    print(ascii_lines[0])
    for line in ascii_lines[1:]:
        print(f"   {line}")
    
    print(f"   Positive count: {result}")
    print("✓ Count excludes negative values test passed")


def test_set_full_grid_neighborhoods_overlapping():
    """Test that overlapping neighborhoods are counted correctly (no double counting)"""
    # Create grid with two seeds that are close together (within L=2 of each other)
    grid = FullGrid((8, 8))
    seeds = [(2, 2), (3, 3)]  # Seeds at (2,2) and (3,3) - Manhattan distance = 2
    grid.set_neighborhoods(seeds, max_distance=2, target_value=2)
    
    # Count should count all positive values, not double-count overlapping regions
    count = grid.count_positive_values()
    
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
    print("TEST: set_full_grid_neighborhoods overlapping (L=2, close seeds)")
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


def test_set_full_grid_neighborhoods_non_overlapping():
    """Test two seeds with Manhattan distance > L (non-overlapping neighborhoods)"""
    # Create grid with two seeds that are far apart (distance > L)
    grid = FullGrid((10, 10))
    seeds = [(2, 2), (7, 7)]  # Seeds at (2,2) and (7,7) - Manhattan distance = 10
    grid.set_neighborhoods(seeds, max_distance=2, target_value=2)
    
    # Count should count positive-valued cells from both neighborhoods
    count = grid.count_positive_values()
    
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
    print("TEST: set_full_grid_neighborhoods non-overlapping (L=2, far seeds)")
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
    # Create a SparseGrid
    sparse = SparseGrid(100, 100, [(10, 10), (20, 20)], L=3)
    assert sparse.num_rows == 100
    assert sparse.num_cols == 100
    assert len(sparse.locations) == 2
    
    # Count should work with L=3
    count = sparse.count_positive_values()
    assert count > 0  # Should have some cells within L=3 of seeds
    
    print()
    print("=" * 70)
    print("TEST: SparseGrid.class")
    print("=" * 70)
    print(f"   Grid dimensions: {sparse.num_rows}x{sparse.num_cols}")
    print(f"   Seed locations: {sparse.locations}")
    print(f"   L: {sparse.L}")
    print(f"   Count: {count}")
    print("✓ SparseGrid class test passed")


def test_hardware_detection():
    """Test that hardware detection works and at least NumPy is available"""
    print("Testing hardware detection...")
    accelerators = detect_available_accelerators()
    
    # At minimum, NumPy should be available
    assert 'numpy' in accelerators, "NumPy should always be available"
    assert accelerators['numpy'] == True, "NumPy should be True"
    
    print(f"  NumPy: {'✓ Available' if accelerators['numpy'] else '✗ Not available'}")
    print(f"  PyTorch: {'✓ Available' if accelerators.get('pytorch_cuda') else '✗ Not available'}")
    print(f"  JAX: {'✓ Available' if accelerators.get('jax_gpu') else '✗ Not available'}")
    print("✓ Hardware detection test passed")


def test_counting_performance_small():
    """Test counting performance on small array"""
    print("Testing counting performance (small array)...")
    array = np.random.randint(0, 10, size=(100, 100))
    
    start = time.perf_counter()
    grid = FullGrid(array)
    result = grid.count_positive_values()
    elapsed = time.perf_counter() - start
    
    assert result > 0, "Should count some positive values"
    assert elapsed < 0.1, "Should be fast (<100ms)"
    
    print(f"  Count: {result} in {elapsed*1000:.4f}ms")
    print("✓ Small array performance test passed")


def test_manhattan_performance_comparison():
    """Test that direct counting is faster than grid creation"""
    print("=" * 70)
    print("TEST: count_sparse_grid vs set_full_grid + count_full_grid performance")
    print("=" * 70)
    print("Testing Manhattan neighborhood performance...")
    
    seeds = [(10, 10), (30, 30), (50, 50)]
    L = 3
    
    # Direct counting
    times_direct = []
    for _ in range(3):
        start = time.perf_counter()
        sparse = SparseGrid(100, 100, seeds, L=L)
        count_direct = sparse.count()
        times_direct.append(time.perf_counter() - start)
    
    avg_direct = np.mean(times_direct)
    
    # Grid creation + count
    times_grid = []
    for _ in range(3):
        grid = FullGrid((100, 100))
        start = time.perf_counter()
        grid.set_neighborhoods(seeds, max_distance=L, target_value=2)
        count_grid = grid.count()
        times_grid.append(time.perf_counter() - start)
    
    avg_grid = np.mean(times_grid)
    
    print(f"  Direct counting: {avg_direct*1000:.4f}ms -> {count_direct} cells")
    print(f"  Grid + counting: {avg_grid*1000:.4f}ms -> {count_grid} cells")
    
    # Both should give same count
    assert count_direct == count_grid, "Counts should match"
    
    print(f"  Direct counting is {avg_grid/avg_direct:.2f}x faster")
    print("✓ Manhattan performance comparison test passed")


def run_all_tests():
    """Run all tests"""
    print("Running tests for 2D array positive value counting...")
    print("-" * 50)
    
    try:
        test_set_full_grid_basic()
        test_set_full_grid_empty()
        test_set_full_grid_out_of_bounds()
        test_set_full_grid_single()
        test_set_full_grid_multiple()
        test_set_full_grid_modify_existing()
        test_set_full_grid_from_sparse_locations()
        test_count_full_grid_basic()
        test_count_full_grid_all_zeros()
        test_count_full_grid_all_nonzero()
        test_count_full_grid_with_set_full_grid()
        test_count_full_grid_excludes_negatives()
        test_set_full_grid_neighborhoods_overlapping()
        test_set_full_grid_neighborhoods_non_overlapping()
        test_sparse_grid_class()
        
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

