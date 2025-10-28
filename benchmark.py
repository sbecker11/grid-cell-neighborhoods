"""
Benchmark script to compare different summation approaches.

This script compares the performance of:
- Naive Python loop
- NumPy (CPU)
- PyTorch (if available)
- JAX (if available)
"""

import time
import numpy as np
from sum_2d_array import intelligent_sum_2d_array, sum_only_positive, detect_available_accelerators


def naive_sum_2d(array):
    """Naive Python implementation for comparison."""
    total = 0
    for row in array:
        for val in row:
            total += val
    return total


def naive_positive_sum_2d(array):
    """Naive Python implementation for positive elements only."""
    total = 0
    for row in array:
        for val in row:
            if val > 0:
                total += val
    return total


def numpy_sum_2d(array):
    """NumPy implementation."""
    return np.sum(array)


def numpy_positive_sum_2d(array):
    """NumPy implementation for positive elements."""
    arr = np.asarray(array)
    return np.sum(arr[arr > 0])


def benchmark(name, func, *args):
    """Run a benchmark and report timing."""
    times = []
    iterations = 5
    
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    print(f"  {name}: {avg_time:.4f}s (avg over {iterations} runs) -> result: {result:,.2f}")
    return avg_time, result


if __name__ == "__main__":
    print("=" * 70)
    print("2D Array Summation Benchmark")
    print("=" * 70)
    
    # Detect available hardware
    accelerators = detect_available_accelerators()
    print(f"Hardware: {accelerators}")
    print()
    
    # Test sizes
    test_sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    for H, W in test_sizes:
        print(f"\n{'=' * 70}")
        print(f"Testing with array size: {H}x{W}")
        print(f"{'=' * 70}")
        
        # Generate test data
        np.random.seed(42)
        test_array = np.random.rand(H, W) * 100
        test_list = test_array.tolist()
        
        print("\n--- Sum All Elements ---")
        
        # Naive Python
        if H * W <= 100000:  # Only run on smaller arrays
            _, naive_result = benchmark("Naive Python", naive_sum_2d, test_list)
        
        # NumPy
        _, numpy_result = benchmark("NumPy CPU", numpy_sum_2d, test_array)
        
        # Intelligent function
        _, intelligent_result = benchmark("Intelligent Sum", intelligent_sum_2d_array, test_array)
        
        # Verify results match
        if H * W <= 100000:
            assert abs(naive_result - numpy_result) < 0.01, "Results don't match!"
        assert abs(numpy_result - intelligent_result) < 0.01, "Results don't match!"
        
        print("\n--- Sum Only Positive Elements ---")
        
        # Add some negative values
        test_array_mixed = test_array - 50  # Some will be negative
        test_list_mixed = test_array_mixed.tolist()
        
        # Naive Python
        if H * W <= 100000:
            _, naive_positive = benchmark("Naive Python (+ only)", naive_positive_sum_2d, test_list_mixed)
        
        # NumPy
        _, numpy_positive = benchmark("NumPy CPU (+ only)", numpy_positive_sum_2d, test_array_mixed)
        
        # Intelligent function
        _, intelligent_positive = benchmark("Intelligent Sum (+ only)", intelligent_sum_2d_array, test_array_mixed, True)
        
        # Verify results match
        if H * W <= 100000:
            assert abs(naive_positive - numpy_positive) < 0.01, "Results don't match!"
        assert abs(numpy_positive - intelligent_positive) < 0.01, "Results don't match!"
    
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    print("\nTip: Install PyTorch or JAX with GPU support for even better performance on large arrays.")

