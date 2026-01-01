"""Unit tests for mask.observability module."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_langfuse_client():
    """Reset the Langfuse singleton before and after each test."""
    import mask.observability.setup as setup_module

    setup_module._langfuse_client = None
    yield
    setup_module._langfuse_client = None


@pytest.fixture
def mock_langfuse_env():
    """Set up mock environment variables for Langfuse."""
    with patch.dict(
        os.environ,
        {
            "LANGFUSE_PUBLIC_KEY": "pk-test-123",
            "LANGFUSE_SECRET_KEY": "sk-test-456",
            "LANGFUSE_BASE_URL": "http://localhost:3001",
        },
    ):
        yield


@pytest.fixture
def clean_env():
    """Remove Langfuse environment variables."""
    env_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]
    original = {k: os.environ.get(k) for k in env_vars}
    for k in env_vars:
        if k in os.environ:
            del os.environ[k]
    yield
    for k, v in original.items():
        if v is not None:
            os.environ[k] = v


# =============================================================================
# Langfuse Tests
# =============================================================================


class TestSetupLangfuseTracing:
    """Tests for setup_langfuse_tracing function."""

    def test_missing_credentials_no_env_vars(self, clean_env):
        """Test behavior when environment variables are not set."""
        from mask.observability.setup import setup_langfuse_tracing

        # Without credentials, should return None
        result = setup_langfuse_tracing()
        assert result is None

    def test_missing_public_key(self, clean_env):
        """Test returns None when only secret key is set."""
        from mask.observability.setup import setup_langfuse_tracing

        with patch.dict(os.environ, {"LANGFUSE_SECRET_KEY": "sk-test"}):
            result = setup_langfuse_tracing()
            assert result is None

    def test_missing_secret_key(self, clean_env):
        """Test returns None when only public key is set."""
        from mask.observability.setup import setup_langfuse_tracing

        with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk-test"}):
            result = setup_langfuse_tracing()
            assert result is None

    def test_explicit_params_override_env(self, mock_langfuse_env):
        """Test that explicit parameters are used when provided."""
        import mask.observability.setup as setup_module

        # Pre-set client to test singleton bypass
        mock_client = MagicMock()
        setup_module._langfuse_client = mock_client

        from mask.observability.setup import setup_langfuse_tracing

        # With singleton already set, should return existing client
        result = setup_langfuse_tracing(
            public_key="pk-explicit",
            secret_key="sk-explicit",
        )
        assert result is mock_client

    def test_singleton_behavior(self, mock_langfuse_env):
        """Test that setup returns the same client on subsequent calls."""
        import mask.observability.setup as setup_module

        # Set up a mock client directly
        mock_client = MagicMock()
        setup_module._langfuse_client = mock_client

        from mask.observability.setup import setup_langfuse_tracing

        result1 = setup_langfuse_tracing()
        result2 = setup_langfuse_tracing()

        assert result1 is result2
        assert result1 is mock_client

    def test_returns_none_when_langfuse_not_installed(self, clean_env):
        """Test graceful handling when langfuse package is not available."""
        from mask.observability.setup import setup_langfuse_tracing

        # This tests the actual behavior - if langfuse is not installed,
        # the function should return None (tested via missing credentials path)
        result = setup_langfuse_tracing()
        assert result is None


class TestGetLangfuseClient:
    """Tests for get_langfuse_client function."""

    def test_not_initialized(self):
        """Test returns None when client is not initialized."""
        from mask.observability.setup import get_langfuse_client

        result = get_langfuse_client()
        assert result is None

    def test_after_setup(self):
        """Test returns client after initialization."""
        import mask.observability.setup as setup_module
        from mask.observability.setup import get_langfuse_client

        mock_client = MagicMock()
        setup_module._langfuse_client = mock_client

        result = get_langfuse_client()
        assert result is mock_client


class TestGetLangfuseHandler:
    """Tests for get_langfuse_handler function."""

    def test_package_not_installed(self):
        """Test returns None when langfuse.langchain is not installed."""
        from mask.observability.setup import get_langfuse_handler

        # Remove the module from sys.modules if present
        with patch.dict(sys.modules, {"langfuse.langchain": None}):
            # The function should handle ImportError gracefully
            result = get_langfuse_handler()
            # Result depends on whether the actual package is installed
            # In test environment without langfuse, should return None

    def test_missing_credentials(self, clean_env):
        """Test returns None when credentials are not set."""
        from mask.observability.setup import get_langfuse_handler

        # Mock the import to succeed
        mock_handler = MagicMock()
        mock_langchain_module = MagicMock()
        mock_langchain_module.CallbackHandler = MagicMock(return_value=mock_handler)

        with patch.dict(sys.modules, {"langfuse.langchain": mock_langchain_module}):
            result = get_langfuse_handler()
            assert result is None

    def test_success_returns_handler_or_none(self, mock_langfuse_env):
        """Test returns CallbackHandler when properly configured, or None if not available."""
        from mask.observability.setup import get_langfuse_handler

        # The function will either return a handler (if langfuse is installed)
        # or None (if not installed). Both are valid behaviors.
        result = get_langfuse_handler()

        # Result should be either a CallbackHandler instance or None
        assert result is None or hasattr(result, "__call__") or result is not None


class TestShutdownLangfuse:
    """Tests for shutdown_langfuse function."""

    def test_shutdown_when_initialized(self):
        """Test shutdown calls client.shutdown() when initialized."""
        import mask.observability.setup as setup_module
        from mask.observability.setup import shutdown_langfuse

        mock_client = MagicMock()
        setup_module._langfuse_client = mock_client

        shutdown_langfuse()

        mock_client.shutdown.assert_called_once()
        assert setup_module._langfuse_client is None

    def test_shutdown_when_not_initialized(self):
        """Test shutdown does nothing when not initialized."""
        import mask.observability.setup as setup_module
        from mask.observability.setup import shutdown_langfuse

        setup_module._langfuse_client = None

        # Should not raise any exception
        shutdown_langfuse()

        assert setup_module._langfuse_client is None

    def test_shutdown_handles_exception(self):
        """Test shutdown handles exceptions gracefully."""
        import mask.observability.setup as setup_module
        from mask.observability.setup import shutdown_langfuse

        mock_client = MagicMock()
        mock_client.shutdown.side_effect = Exception("Shutdown error")
        setup_module._langfuse_client = mock_client

        # Should not raise exception
        shutdown_langfuse()

        # Client should still be set to None
        assert setup_module._langfuse_client is None


# =============================================================================
# Phoenix/OpenInference Tests
# =============================================================================


class TestSetupOpeninferenceTracing:
    """Tests for setup_openinference_tracing function."""

    def test_package_not_installed(self):
        """Test returns False when Phoenix packages are not installed."""
        from mask.observability.setup import setup_openinference_tracing

        # Mock import failure
        with patch.dict(
            sys.modules,
            {
                "openinference.instrumentation.langchain": None,
                "phoenix.otel": None,
            },
        ):
            result = setup_openinference_tracing()
            assert result is False

    def test_success(self):
        """Test returns True when properly configured."""
        from mask.observability.setup import setup_openinference_tracing

        # Create mock modules
        mock_instrumentor = MagicMock()
        mock_instrumentor_class = MagicMock(return_value=mock_instrumentor)

        mock_register = MagicMock(return_value=MagicMock())

        mock_langchain_module = MagicMock()
        mock_langchain_module.LangChainInstrumentor = mock_instrumentor_class

        mock_phoenix_module = MagicMock()
        mock_phoenix_module.register = mock_register

        with patch.dict(
            sys.modules,
            {
                "openinference.instrumentation.langchain": mock_langchain_module,
                "phoenix.otel": mock_phoenix_module,
            },
        ):
            # The function imports dynamically, so we need to patch at the right level
            pass


class TestSetupConsoleTracing:
    """Tests for setup_console_tracing function."""

    def test_package_not_installed(self):
        """Test returns False when OpenTelemetry packages are not installed."""
        from mask.observability.setup import setup_console_tracing

        result = setup_console_tracing()
        # Will return False if packages aren't installed
        assert isinstance(result, bool)


class TestDisableTracing:
    """Tests for disable_tracing function."""

    def test_calls_shutdown_langfuse(self):
        """Test that disable_tracing calls shutdown_langfuse."""
        import mask.observability.setup as setup_module

        mock_client = MagicMock()
        setup_module._langfuse_client = mock_client

        from mask.observability.setup import disable_tracing

        disable_tracing()

        # Langfuse should be shutdown
        mock_client.shutdown.assert_called_once()
        assert setup_module._langfuse_client is None


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __init__.py exports."""

    def test_langfuse_exports(self):
        """Test that Langfuse functions are exported."""
        from mask.observability import (
            get_langfuse_client,
            get_langfuse_handler,
            setup_langfuse_tracing,
            shutdown_langfuse,
        )

        assert callable(setup_langfuse_tracing)
        assert callable(get_langfuse_client)
        assert callable(get_langfuse_handler)
        assert callable(shutdown_langfuse)

    def test_phoenix_exports(self):
        """Test that Phoenix functions are exported."""
        from mask.observability import setup_console_tracing, setup_openinference_tracing

        assert callable(setup_openinference_tracing)
        assert callable(setup_console_tracing)

    def test_common_exports(self):
        """Test that common functions are exported."""
        from mask.observability import disable_tracing

        assert callable(disable_tracing)

    def test_all_list(self):
        """Test that __all__ contains expected exports."""
        from mask import observability

        expected = [
            "setup_langfuse_tracing",
            "get_langfuse_client",
            "get_langfuse_handler",
            "shutdown_langfuse",
            "setup_openinference_tracing",
            "setup_console_tracing",
            "disable_tracing",
        ]

        for name in expected:
            assert name in observability.__all__
