#!/usr/bin/env python3
"""
Simple test script for the NumPy-only accelerated_matrix module.
Demonstrates cross-platform usage on macOS, Windows, and Linux.
"""

from accelerated_matrix import (
    DenseMatrix,
    SparseMatrix,
    create_dense_matrix,
    create_sparse_matrix,
    detect_platform,
    get_platform_info
)

def test_platform_detection():
    """Test platform detection."""
    print("=" * 70)
    print("Platform Detection")
    print("=" * 70)
    info = get_platform_info()
    for key, value in info.items():
        if key == 'optimization_hints':
            print(f"\nOptimization Hints:")
            for hint_key, hint_value in value.items():
                print(f"  {hint_key}: {hint_value}")
        else:
            print(f"{key}: {value}")
    print()

def test_dense_matrix():
    """Test dense matrix operations."""
    print("=" * 70)
    print("Dense Matrix Operations")
    print("=" * 70)
    
    # Create matrix
    matrix = create_dense_matrix(5, 5, initial_value=0)
    print(f"Created: {matrix}")
    
    # Set specific locations
    matrix.set_locations([(0, 0), (1, 1), (2, 2), (4, 4)], value=2)
    print(f"After setting locations: count_positive() = {matrix.count_positive()}")
    
    # Set neighborhoods
    matrix.set_neighborhoods([(1, 1)], L=2, value=3)
    print(f"After setting neighborhood (L=2): count_positive() = {matrix.count_positive()}")
    print(f"Sum: {matrix.sum()}")
    
    # Convert to NumPy array
    arr = matrix.to_numpy()
    print(f"NumPy array shape: {arr.shape}, dtype: {arr.dtype}")
    print()

def test_sparse_matrix():
    """Test sparse matrix operations."""
    print("=" * 70)
    print("Sparse Matrix Operations")
    print("=" * 70)
    
    # Create sparse matrix
    sparse = create_sparse_matrix(10, 10)
    print(f"Created: {sparse}")
    
    # Set Manhattan neighborhoods
    sparse.set_manhattan_neighborhoods([(5, 5), (2, 2)], L=3, value=2)
    print(f"After setting neighborhoods (L=3): {len(sparse.locations)} locations")
    print(f"Positive count: {sparse.count_positive()}")
    
    # Convert to dense
    dense = sparse.to_dense()
    print(f"Converted to dense: {dense}")
    print(f"Dense positive count: {dense.count_positive()}")
    print()

if __name__ == "__main__":
    import sys
    
    print("\nTesting accelerated_matrix module (NumPy-only, cross-platform)\n")
    
    try:
        test_platform_detection()
        test_dense_matrix()
        test_sparse_matrix()
        
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print("=" * 70)
        print(f"✗ Test failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)

