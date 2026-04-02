"""Tests for semantic-forge main entry point."""

import sys
import pytest
from unittest.mock import patch, AsyncMock
from io import StringIO


class TestMainCLI:
    """Test cases for CLI argument parsing."""

    def test_list_concepts_displays_concepts(self):
        """Test --list-concepts shows all concepts."""
        from semantic_forge.main import main

        with patch("sys.argv", ["semantic-forge", "--list-concepts"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = main()

                assert result == 0
                output = mock_stdout.getvalue()
                assert "Available Behavioral Concepts" in output
                assert "temporal_trust" in output

    def test_show_concept_displays_details(self):
        """Test --concept shows concept details."""
        from semantic_forge.main import main

        with patch("sys.argv", ["semantic-forge", "--concept", "temporal_trust"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = main()

                assert result == 0
                output = mock_stdout.getvalue()
                assert "Concept: Temporal Trust" in output
                assert "Core Statement" in output

    def test_show_concept_not_found(self):
        """Test --concept with invalid ID shows error."""
        from semantic_forge.main import main

        with patch("sys.argv", ["semantic-forge", "--concept", "invalid_concept_id"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = main()

                assert result == 0
                output = mock_stdout.getvalue()
                assert "Concept not found" in output

    def test_default_shows_help(self):
        """Test default behavior shows help."""
        from semantic_forge.main import main

        with patch("sys.argv", ["semantic-forge"]):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                result = main()

                assert result == 0
                output = mock_stdout.getvalue()
                assert "usage: semantic-forge" in output.lower()

    def test_server_flag_accepted(self):
        """Test --server flag is accepted (doesn't crash)."""
        from semantic_forge.main import main

        # Mock asyncio.run to avoid actually starting server
        with patch("semantic_forge.main.asyncio.run") as mock_run:
            with patch("sys.argv", ["semantic-forge", "--server"]):
                result = main()

                assert result == 0
                # Verify run_server was scheduled
                mock_run.assert_called_once()


class TestRunServer:
    """Test cases for run_server function."""

    def test_run_server_signature_no_host_port(self):
        """Test run_server no longer accepts host/port parameters (stdio-only)."""
        from semantic_forge.main import run_server
        import inspect

        sig = inspect.signature(run_server)
        params = list(sig.parameters.keys())

        # Verify no host/port parameters
        assert "host" not in params, "run_server should not accept host parameter"
        assert "port" not in params, "run_server should not accept port parameter"
        # Only 'self' for methods, or no params for async function
        assert len(params) == 0, f"run_server should have no parameters, got {params}"

    def test_main_rejects_host_port_arguments(self):
        """Test main() rejects --host and --port arguments (stdio-only)."""
        from semantic_forge.main import main

        # argparse raises SystemExit for unrecognized arguments
        with patch("sys.argv", ["semantic-forge", "--server", "--host", "0.0.0.0"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 2  # argparse error code

    def test_server_flag_works_without_host_port(self):
        """Test --server flag works without host/port arguments."""
        from semantic_forge.main import main

        with patch("semantic_forge.main.asyncio.run") as mock_run:
            mock_run.return_value = None

            with patch("sys.argv", ["semantic-forge", "--server"]):
                result = main()

                assert result == 0
                mock_run.assert_called_once()
