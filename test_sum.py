"""
Unit tests for 2D array positive value counting functions.
"""

import numpy as np
from sum_2d_array import detect_available_accelerators, set_locations_to_one, create_sparse_grid, count_nonzero_2d_array


def test_set_locations_basic():
    """Test basic set_locations_to_one functionality (now sets to 2)"""
    result = set_locations_to_one((3, 3), [(0, 0), (1, 1), (2, 2)])
    expected = np.zeros((3, 3), dtype=np.int32)
    expected[0, 0] = 2
    expected[1, 1] = 2
    expected[2, 2] = 2
    assert np.array_equal(result, expected)
    print("✓ Basic set_locations test passed")


def test_set_locations_empty():
    """Test with empty locations list"""
    result = set_locations_to_one((5, 5), [])
    expected = np.zeros((5, 5), dtype=np.int32)
    assert np.array_equal(result, expected)
    print("✓ Empty locations test passed")


def test_set_locations_out_of_bounds():
    """Test with out-of-bounds locations (should be ignored)"""
    result = set_locations_to_one((3, 3), [(0, 0), (10, 10), (-1, 0), (1, 1)])
    expected = np.zeros((3, 3), dtype=np.int32)
    expected[0, 0] = 2
    expected[1, 1] = 2
    assert np.array_equal(result, expected)
    print("✓ Out-of-bounds locations test passed")


def test_set_locations_single():
    """Test with single location"""
    result = set_locations_to_one((5, 5), [(2, 3)])
    assert result[2, 3] == 2
    assert np.sum(result) == 2
    print("✓ Single location test passed")


def test_set_locations_multiple():
    """Test with multiple locations"""
    locations = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]
    result = set_locations_to_one((5, 5), locations)
    assert np.sum(result) == len(locations) * 2  # Each location set to 2
    for r, c in locations:
        assert result[r, c] == 2
    print("✓ Multiple locations test passed")


def test_set_locations_modify_existing():
    """Test modifying existing array"""
    existing = np.zeros((3, 3), dtype=np.int32)
    existing[0, 0] = 5  # Set an existing value
    result = set_locations_to_one(existing, [(1, 1)])
    assert result[0, 0] == 5  # Original value preserved
    assert result[1, 1] == 2  # New location set to 2
    assert np.sum(result) == 7
    print("✓ Modify existing array test passed")


def test_create_sparse_grid():
    """Test the convenience function"""
    result = create_sparse_grid(4, 4, [(0, 0), (3, 3)])
    assert result.shape == (4, 4)
    assert result[0, 0] == 2
    assert result[3, 3] == 2
    assert np.sum(result) == 4
    print("✓ Create sparse grid test passed")


def test_set_locations_combine_with_count():
    """Test integration with count function"""
    grid = set_locations_to_one((5, 5), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
    count = count_nonzero_2d_array(grid)
    assert count == 5
    print("✓ Integration with count function test passed")


def test_count_nonzero_basic():
    """Test basic count_nonzero functionality (counts only positive values)"""
    array = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    result = count_nonzero_2d_array(array)
    assert result == 5  # All are positive
    print("✓ Basic count_nonzero test passed")


def test_count_nonzero_all_zeros():
    """Test count_nonzero with all zeros"""
    array = np.zeros((3, 3))
    result = count_nonzero_2d_array(array)
    assert result == 0  # No positive values
    print("✓ Count non-zero all zeros test passed")


def test_count_nonzero_all_nonzero():
    """Test count_nonzero with all positive values"""
    array = np.ones((3, 3))
    result = count_nonzero_2d_array(array)
    assert result == 9  # All are positive
    print("✓ Count non-zero all positive test passed")


def test_count_nonzero_with_set_locations():
    """Test count_nonzero with set_locations_to_one (values set to 2 = positive)"""
    grid = set_locations_to_one((5, 5), [(0, 0), (1, 1), (2, 2)])
    count = count_nonzero_2d_array(grid)
    assert count == 3  # 2 is positive
    print("✓ Count non-zero with set_locations test passed")


def test_count_nonzero_with_negatives():
    """Test that negative values are excluded"""
    array = np.array([[1, -1, 0, 2], [0, -3, 1, 0], [-5, 0, 3, 0]])
    result = count_nonzero_2d_array(array)
    assert result == 4  # Only positive: 1, 2, 1, 3 (negatives and zeros excluded)
    print("✓ Count excludes negative values test passed")


def run_all_tests():
    """Run all tests"""
    print("Running tests for 2D array positive value counting...")
    print("-" * 50)
    
    try:
        test_set_locations_basic()
        test_set_locations_empty()
        test_set_locations_out_of_bounds()
        test_set_locations_single()
        test_set_locations_multiple()
        test_set_locations_modify_existing()
        test_create_sparse_grid()
        test_set_locations_combine_with_count()
        test_count_nonzero_basic()
        test_count_nonzero_all_zeros()
        test_count_nonzero_all_nonzero()
        test_count_nonzero_with_set_locations()
        test_count_nonzero_with_negatives()
        
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

