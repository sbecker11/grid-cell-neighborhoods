"""
Demo script for set_locations_to_one function.

This demonstrates the efficient way to set specific locations in a 2D array to 1.
"""

import numpy as np
from count_2d_array import set_locations_to_one, create_sparse_grid, intelligent_sum_2d_array

print("=" * 70)
print("Demo: Setting Specific Locations in 2D Arrays")
print("=" * 70)
print()

# Example 1: Basic usage - create a grid with specific locations set to 1
print("Example 1: Basic usage - diagonal pattern")
locations = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
grid = set_locations_to_one((5, 5), locations)
print(f"Locations: {locations}")
print("Grid:")
print(grid)
print(f"Total active cells: {intelligent_sum_2d_array(grid)}")
print()

# Example 2: Creating a pattern (cross shape)
print("Example 2: Creating a cross pattern")
rows = [(2, i) for i in range(5)]  # Horizontal line
cols = [(i, 2) for i in range(5)]  # Vertical line
cross_locations = list(set(rows + cols))  # Remove duplicate center
grid = set_locations_to_one((5, 5), cross_locations)
print("Cross pattern:")
print(grid)
print()

# Example 3: Modifying existing array
print("Example 3: Modifying existing array")
existing = np.zeros((4, 4), dtype=np.int32)
existing[0, 0] = 5  # Set some existing value
existing[1, 1] = 3

new_locations = [(2, 2), (3, 3)]
modified = set_locations_to_one(existing, new_locations)
print("Original array:")
print(existing)
print("Modified array (new locations set to 1):")
print(modified)
print(f"Note: Original values preserved: {modified[0, 0]} and {modified[1, 1]}")
print()

# Example 4: Handle out-of-bounds indices gracefully
print("Example 4: Handling out-of-bounds indices")
locations_with_bounds = [(0, 0), (1, 1), (2, 2), (-1, 0), (10, 10)]
grid = set_locations_to_one((3, 3), locations_with_bounds)
print(f"Input locations (some out of bounds): {locations_with_bounds}")
print("Result (out-of-bounds ignored):")
print(grid)
print()

# Example 5: Large sparse grid
print("Example 5: Large sparse grid")
# Create a 100x100 grid with random sparse locations
np.random.seed(42)
num_locations = 50
random_locations = [(np.random.randint(0, 100), np.random.randint(0, 100)) 
                    for _ in range(num_locations)]
grid = set_locations_to_one((100, 100), random_locations)
print(f"Grid size: 100x100")
print(f"Total locations set: {len(random_locations)}")
print(f"Active cells in grid: {intelligent_sum_2d_array(grid):.0f}")
print(f"Percentage filled: {intelligent_sum_2d_array(grid) / (100 * 100) * 100:.2f}%")
print()

# Example 6: Convenience function
print("Example 6: Using convenience function create_sparse_grid")
grid = create_sparse_grid(3, 3, [(0, 0), (1, 1), (2, 2)])
print("Grid created with convenience function:")
print(grid)
print()

# Example 7: Real-world use case - neighborhood grid
print("Example 7: Neighborhood cell grid")
print("Simulating a grid where certain cells are 'active'")
np.random.seed(123)
# Create random active locations
active_cells = [(np.random.randint(0, 10), np.random.randint(0, 10)) 
                for _ in range(15)]
neighborhood = set_locations_to_one((10, 10), active_cells)
print("Neighborhood grid (10x10, 15 active cells):")
print(neighborhood)
print(f"Active cell locations: {active_cells[:5]}...")  # Show first 5
print()

# Example 8: Performance with many locations
print("Example 8: Performance with many locations")
import time
sizes = [(100, 100), (500, 500), (1000, 1000)]
for h, w in sizes:
    # Generate 10% of cells as active
    num_active = int(h * w * 0.1)
    random_locations = [(np.random.randint(0, h), np.random.randint(0, w)) 
                        for _ in range(num_active)]
    
    start = time.perf_counter()
    grid = set_locations_to_one((h, w), random_locations)
    end = time.perf_counter()
    
    total = intelligent_sum_2d_array(grid)
    print(f"Grid {h}x{w}: Set {num_active:,} locations in {end-start:.4f}s "
          f"-> Sum: {total:.0f}")
print()

print("=" * 70)
print("Summary:")
print("=" * 70)
print("The set_locations_to_one() function:")
print("  • Efficiently sets multiple locations to 1 in a 2D array")
print("  • Handles out-of-bounds indices gracefully")
print("  • Can create new arrays or modify existing ones")
print("  • Uses vectorized NumPy operations for speed")
print("  • Works with the intelligent_sum_2d_array() for complete workflows")
print()
print("Use cases:")
print("  • Grid-based algorithms and simulations")
print("  • Sparse matrix creation")
print("  • Neighborhood/cellular automata")
print("  • Pathfinding and graph algorithms")
print()

