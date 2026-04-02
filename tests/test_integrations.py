"""Tests for external MCP integrations."""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from semantic_forge.integrations import SemanticKinematicsClient, PromptPrixClient


class TestSemanticKinematicsClient:
    """Test cases for SemanticKinematicsClient."""

    def test_parse_endpoint_simple_command(self):
        """Test parsing a simple command endpoint."""
        client = SemanticKinematicsClient("semantic-kinematics-mcp")
        command, args, env = client._parse_endpoint("semantic-kinematics-mcp")

        assert command == "semantic-kinematics-mcp"
        assert args == []
        assert env is None

    def test_parse_endpoint_with_args(self):
        """Test parsing endpoint with comma-separated args."""
        client = SemanticKinematicsClient("my-command,arg1,arg2")
        command, args, env = client._parse_endpoint("my-command,arg1,arg2")

        assert command == "my-command"
        assert args == ["arg1", "arg2"]
        assert env is None

    def test_parse_endpoint_docker_format(self):
        """Test parsing docker: prefix format."""
        client = SemanticKinematicsClient("docker:semantic-kinematics-mcp")
        command, args, env = client._parse_endpoint("docker:semantic-kinematics-mcp")

        assert command == "docker"
        assert "run" in args
        assert "-i" in args
        assert "--rm" in args
        assert "semantic-kinematics-mcp" in args
        assert env is None

    def test_parse_endpoint_docker_with_options(self):
        """Test parsing docker format with additional options."""
        client = SemanticKinematicsClient("docker:run,-i,--rm,network=host,image-name")
        command, args, env = client._parse_endpoint("docker:run,-i,--rm,network=host,image-name")

        assert command == "docker"
        # When "run" is first arg, use args as-is without prepending defaults
        assert args == ["run", "-i", "--rm", "network=host", "image-name"]
        assert env is None

    def test_endpoint_attribute_set_correctly(self):
        """Test that endpoint attribute is set from constructor."""
        client = SemanticKinematicsClient("custom-endpoint")
        assert client.endpoint == "custom-endpoint"

    def test_endpoint_default_value(self):
        """Test that endpoint defaults to semantic-kinematics-mcp."""
        client = SemanticKinematicsClient()
        assert client.endpoint == "semantic-kinematics-mcp"

    @pytest.mark.asyncio
    async def test_initialize_uses_endpoint(self):
        """Test that initialize() uses the endpoint parameter."""
        client = SemanticKinematicsClient("custom-mcp-command")

        mock_read = AsyncMock()
        mock_write = AsyncMock()

        # Create proper async context manager mock for stdio_client
        @asynccontextmanager
        async def mock_stdio_async(params):
            # Verify the params were passed correctly
            assert params.command == "custom-mcp-command"
            yield (mock_read, mock_write)

        with patch("semantic_forge.integrations.stdio_client", side_effect=mock_stdio_async):
            with patch("semantic_forge.integrations.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                # Mock both __aenter__ and the instance itself for enter_async_context
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_class.return_value = mock_session

                await client.initialize()

                assert client._initialized is True
                await client.close()

    @pytest.mark.asyncio
    async def test_initialize_uses_docker_endpoint(self):
        """Test that initialize() correctly handles docker: prefix."""
        client = SemanticKinematicsClient("docker:my-image")

        mock_read = AsyncMock()
        mock_write = AsyncMock()

        @asynccontextmanager
        async def mock_stdio_async(params):
            # Verify docker command is used
            assert params.command == "docker"
            assert "my-image" in params.args
            yield (mock_read, mock_write)

        with patch("semantic_forge.integrations.stdio_client", side_effect=mock_stdio_async):
            with patch("semantic_forge.integrations.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_class.return_value = mock_session

                await client.initialize()

                assert client._initialized is True
                await client.close()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize() is idempotent (safe to call multiple times)."""
        client = SemanticKinematicsClient("test-command")

        mock_read = AsyncMock()
        mock_write = AsyncMock()
        call_count = [0]  # Use list to modify in closure

        @asynccontextmanager
        async def mock_stdio_async(params):
            call_count[0] += 1
            yield (mock_read, mock_write)

        with patch("semantic_forge.integrations.stdio_client", side_effect=mock_stdio_async):
            with patch("semantic_forge.integrations.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_class.return_value = mock_session

                # Call initialize twice
                result1 = await client.initialize()
                result2 = await client.initialize()

                # Should only call stdio_client once
                assert call_count[0] == 1
                assert result1 is True
                assert result2 is True
                await client.close()
                assert result1 is True
                assert result2 is True

    @pytest.mark.asyncio
    async def test_initialize_raises_on_failure(self):
        """Test that initialize() raises RuntimeError on failure."""
        client = SemanticKinematicsClient("failing-command")

        with patch("semantic_forge.integrations.stdio_client") as mock_stdio:
            mock_stdio.side_effect = ConnectionError("Connection refused")

            with pytest.raises(RuntimeError) as exc_info:
                await client.initialize()

            assert "Failed to initialize semantic-kinematics-mcp" in str(exc_info.value)

    def test_client_accepts_backend_config(self):
        """Test that constructor accepts backend config parameters."""
        client = SemanticKinematicsClient(
            endpoint="test-endpoint",
            backend="lmstudio",
            base_url="http://localhost:1234/v1",
            model_name="text-embedding-embeddinggemma-300m",
        )

        assert client.backend_config["backend"] == "lmstudio"
        assert client.backend_config["base_url"] == "http://localhost:1234/v1"
        assert client.backend_config["model_name"] == "text-embedding-embeddinggemma-300m"

    def test_backend_config_defaults_to_none(self):
        """Test that backend config defaults to None when not specified."""
        client = SemanticKinematicsClient("test-endpoint")

        assert client.backend_config["backend"] is None
        assert client.backend_config["base_url"] is None
        assert client.backend_config["model_name"] is None

    @pytest.mark.asyncio
    async def test_ensure_backend_no_config(self):
        """Test _ensure_backend() is a no-op when no backend configured."""
        client = SemanticKinematicsClient("test-endpoint")

        # Should not raise or do anything when no backend config
        await client._ensure_backend()

    @pytest.mark.asyncio
    async def test_ensure_backend_calls_model_load(self):
        """Test _ensure_backend() calls model_load with backend config."""
        client = SemanticKinematicsClient(
            endpoint="test-endpoint",
            backend="lmstudio",
            base_url="http://localhost:1234/v1",
            model_name="embeddinggemma",
        )

        mock_session = AsyncMock()
        client._session = mock_session

        # Mock _call_tool to capture the call
        captured_args = {}

        async def mock_call_tool(name, arguments):
            captured_args[name] = arguments
            return {"status": "loaded", "backend": "lmstudio"}

        client._call_tool = mock_call_tool

        await client._ensure_backend()

        # Verify model_load was called with correct args
        assert "model_load" in captured_args
        assert captured_args["model_load"]["backend"] == "lmstudio"
        assert captured_args["model_load"]["base_url"] == "http://localhost:1234/v1"
        assert captured_args["model_load"]["model_name"] == "embeddinggemma"

    @pytest.mark.asyncio
    async def test_model_status_method(self):
        """Test model_status() calls the correct tool."""
        client = SemanticKinematicsClient("test-endpoint")

        mock_session = AsyncMock()
        client._session = mock_session

        async def mock_call_tool(name, arguments):
            return {"backend": "lmstudio", "model_name": "test", "is_loaded": True}

        client._call_tool = mock_call_tool

        result = await client.model_status()

        assert result["backend"] == "lmstudio"

    @pytest.mark.asyncio
    async def test_model_load_method(self):
        """Test model_load() passes arguments correctly."""
        client = SemanticKinematicsClient("test-endpoint")

        mock_session = AsyncMock()
        client._session = mock_session

        captured_args = {}

        async def mock_call_tool(name, arguments):
            captured_args[name] = arguments
            return {"status": "loaded"}

        client._call_tool = mock_call_tool

        await client.model_load(
            backend="sentence_transformers",
            model_name="all-MiniLM-L6-v2",
        )

        assert "model_load" in captured_args
        assert captured_args["model_load"]["backend"] == "sentence_transformers"
        assert captured_args["model_load"]["model_name"] == "all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_model_unload_method(self):
        """Test model_unload() passes clear_cache argument."""
        client = SemanticKinematicsClient("test-endpoint")

        mock_session = AsyncMock()
        client._session = mock_session

        captured_args = {}

        async def mock_call_tool(name, arguments):
            captured_args[name] = arguments
            return {"status": "unloaded"}

        client._call_tool = mock_call_tool

        await client.model_unload(clear_cache=True)

        assert "model_unload" in captured_args
        assert captured_args["model_unload"]["clear_cache"] is True


class TestPromptPrixClient:
    """Test cases for PromptPrixClient."""

    def test_endpoint_attribute_set_correctly(self):
        """Test that endpoint attribute is set from constructor."""
        client = PromptPrixClient("custom-prompt-prix")
        assert client.endpoint == "custom-prompt-prix"

    def test_endpoint_default_value(self):
        """Test that endpoint defaults to prompt-prix-mcp."""
        client = PromptPrixClient()
        assert client.endpoint == "prompt-prix-mcp"
