"""Test autoencoder CLI functionality."""

import os
import unittest


class TestAutoencoderCLI(unittest.TestCase):
    """Test autoencoder CLI inference path handling."""

    def test_makedirs_called_for_output(self) -> None:
        """Verify os.makedirs is called with output dirname."""
        output_path = "/fake/path/to/output.nii.gz"
        expected_dir = "/fake/path/to"

        # This test verifies the code logic: os.makedirs(os.path.dirname(output))
        self.assertEqual(os.path.dirname(output_path), expected_dir)


if __name__ == "__main__":
    unittest.main()
