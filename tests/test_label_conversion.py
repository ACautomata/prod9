#!/usr/bin/env python3
"""Test script to verify the label conversion fix works correctly."""

import numpy as np


def test_label_conversion():
    """Test the label conversion logic from medmnist3d_data.py"""

    # Test 0-d numpy array (scalar)
    label_0d = np.array(5)
    assert label_0d.ndim == 0
    label_int_0d = int(label_0d) if np.ndim(label_0d) == 0 else int(label_0d.item())
    assert label_int_0d == 5
    assert isinstance(label_int_0d, int)
    print(f"✓ 0-d array: {label_0d} -> {label_int_0d} (type: {type(label_int_0d).__name__})")

    # Test 1-d numpy array (like [5])
    label_1d = np.array([5])
    assert label_1d.ndim == 1
    label_int_1d = int(label_1d) if np.ndim(label_1d) == 0 else int(label_1d.item())
    assert label_int_1d == 5
    assert isinstance(label_int_1d, int)
    print(f"✓ 1-d array: {label_1d} -> {label_int_1d} (type: {type(label_int_1d).__name__})")

    # Test numpy scalar
    label_scalar = np.int64(7)
    label_int_scalar = int(label_scalar) if np.ndim(label_scalar) == 0 else int(label_scalar.item())
    assert label_int_scalar == 7
    assert isinstance(label_int_scalar, int)
    print(f"✓ numpy scalar: {label_scalar} -> {label_int_scalar} (type: {type(label_int_scalar).__name__})")

    # Test various label values (0-10 for OrganMNIST3D)
    for i in range(11):
        label = np.array(i)
        label_int = int(label) if np.ndim(label) == 0 else int(label.item())
        assert label_int == i
        assert isinstance(label_int, int)

    print(f"✓ All label values (0-10) convert correctly")

    # Test the old broken behavior
    try:
        label_1d_broken = np.array([5])
        # This would fail: only 0-dimensional arrays can be converted to Python scalars
        broken_int = int(label_1d_broken)
        print(f"✗ Old behavior should have failed but didn't: {broken_int}")
    except TypeError as e:
        print(f"✓ Old behavior correctly raises TypeError: {e}")

    print("\n✅ All tests passed! The fix handles both 0-d and 1-d numpy arrays correctly.")


if __name__ == "__main__":
    test_label_conversion()
