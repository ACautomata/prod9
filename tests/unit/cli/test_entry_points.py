"""Test CLI entry points."""

import pytest
from unittest.mock import patch


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

    def test_autoencoder_cli_help(self) -> None:
        """Autoencoder CLI should exit cleanly when showing help."""
        from prod9.cli.autoencoder import main
        with patch("sys.argv", ["prod9-train-autoencoder", "--help"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 0

    def test_transformer_cli_help(self) -> None:
        """Transformer CLI should exit cleanly when showing help."""
        from prod9.cli.transformer import main
        with patch("sys.argv", ["prod9-train-transformer", "--help"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 0
