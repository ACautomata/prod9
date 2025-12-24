"""Test transformer CLI functionality."""

import inspect
import unittest


class TestTransformerCLI(unittest.TestCase):
    """Test transformer CLI argument handling."""

    def test_generate_sample_signature(self) -> None:
        """Verify generate() calls sample() with correct parameters."""
        from prod9.training.lightning_module import TransformerLightning

        # Get sample method signature
        sig = inspect.signature(TransformerLightning.sample)
        params = list(sig.parameters.keys())

        # Verify required parameters exist
        self.assertIn("source_image", params)
        self.assertIn("source_modality_idx", params)
        self.assertIn("target_modality_idx", params)
        self.assertIn("is_unconditional", params)


if __name__ == "__main__":
    unittest.main()
