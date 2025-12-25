"""Test CLI entry points."""

import pytest


class TestCLIEntryPoints:
    """Test CLI module entry points are accessible."""

    def test_autoencoder_cli_entry_point(self) -> None:
        """Verify autoencoder CLI main function is callable."""
        from prod9.cli.autoencoder import main
        assert callable(main)

    def test_transformer_cli_entry_point(self) -> None:
        """Verify transformer CLI main function is callable."""
        from prod9.cli.transformer import main
        assert callable(main)
