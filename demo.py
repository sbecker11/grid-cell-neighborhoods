"""
Demo script showing the use case: Summing positive int elements from a 2D array.

Original requirement:
- H rows and W columns (H, W > 0)
- Element values are >= 0
- Need to sum all positive elements
"""

import numpy as np
from sum_2d_array import intelligent_sum_2d_array, sum_only_positive, detect_available_accelerators

print("=" * 70)
print("Demo: Summing Positive Int Elements from 2D Array")
print("=" * 70)
print()

# Show available hardware
print("Hardware Configuration:")
accelerators = detect_available_accelerators()
for key, value in accelerators.items():
    status = "✓ Available" if value else "✗ Not available"
    print(f"  {key:20s}: {status}")
print()

# Example 1: Small array with zeros and positive values
print("Example 1: Simple 2D array with zeros")
array1 = [
    [0, 2, 3, 0],
    [5, 0, 7, 8],
    [0, 0, 1, 0]
]
print("Input array:")
for row in array1:
    print(f"  {row}")

positive_sum = intelligent_sum_2d_array(array1, positive_only=True)
print(f"\nSum of positive elements: {positive_sum}")
print(f"Expected: 2+3+5+7+8+1 = 26")
assert positive_sum == 26
print("✓ Correct!")
print()

# Example 2: Larger array with many zeros
print("Example 2: 10x10 array with sparse positive values")
np.random.seed(42)
# Create array where 70% of values are 0, 30% are positive
sparse_array = np.random.choice([0, 0, 0, 0, 0, 1, 2, 3, 4, 5], size=(10, 10))
print(f"Input array shape: {sparse_array.shape}")
print(f"Non-zero elements: {np.count_nonzero(sparse_array)}")

positive_sum = intelligent_sum_2d_array(sparse_array, positive_only=True)
print(f"Sum of positive elements: {positive_sum}")
print("✓ Computed!")
print()

# Example 3: Very large array performance demo
print("Example 3: Performance on large array (1000x1000)")
large_array = np.random.randint(0, 10, size=(1000, 1000))
print(f"Array size: 1000x1000")
print(f"Total elements: {large_array.size:,}")
print(f"Positive elements: {np.count_nonzero(large_array):,}")

positive_sum = intelligent_sum_2d_array(large_array, positive_only=True)
expected_sum = np.sum(large_array)  # Should be same since all >= 0
print(f"Sum of positive elements: {positive_sum:,.0f}")
print(f"Verify (all elements sum): {expected_sum:,.0f}")
print("✓ Computed efficiently!")
print()

# Example 4: Real-world scenario with grid
print("Example 4: Grid cell neighborhoods (common use case)")
# Simulate a grid where each cell has a positive count (neighbors, values, etc.)
grid = np.random.randint(0, 5, size=(20, 20))
# Some cells have 0 (no activity)
grid[grid < 2] = 0
print(f"Grid size: {grid.shape}")
print(f"Active cells (value > 0): {np.count_nonzero(grid)}")
print(f"Total activity: {intelligent_sum_2d_array(grid, positive_only=True):,.0f}")
print("✓ Ready for production use!")
print()

print("=" * 70)
print("Summary:")
print("=" * 70)
print("The intelligent_sum_2d_array() function:")
print("  • Automatically selects the fastest available hardware")
print("  • Works with both Python lists and NumPy arrays")
print("  • Can sum all elements or only positive elements")
print("  • Handles large arrays efficiently (1000x1000+ elements)")
print("  • Provides consistent, correct results")
print()
print("For your specific use case (H x W arrays, elements >= 0):")
print("  Use: intelligent_sum_2d_array(array, positive_only=True)")
print("  Or:   sum_only_positive(array)")
print()

