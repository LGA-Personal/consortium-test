"""
Tests for VerifiedLayer wrapper.

Tests that VerifiedLayer correctly wraps layers and invokes callbacks
without affecting the layer's output.
"""

import pytest
from unittest.mock import Mock, call
import numpy as np

# Try to import MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from exo.consortium.exo_integration.verified_layer import (
    VerifiedLayer,
    wrap_layers_for_verification,
    MLX_AVAILABLE,
)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestVerifiedLayerWithMLX:
    """Tests for VerifiedLayer with actual MLX layers."""

    def test_output_unchanged(self):
        """Test that VerifiedLayer doesn't change layer output."""
        # Create a simple linear layer
        layer = nn.Linear(64, 64)

        # Create callback that records calls
        calls = []
        def callback(layer_idx: int, hidden_state: mx.array):
            calls.append((layer_idx, hidden_state.shape))

        # Wrap layer
        verified = VerifiedLayer(layer, layer_idx=5, callback=callback)

        # Create input
        x = mx.random.normal((2, 16, 64))

        # Execute both
        expected = layer(x)
        actual = verified(x)

        # Outputs should match
        mx.eval(expected, actual)
        np_expected = np.array(expected)
        np_actual = np.array(actual)

        assert np.allclose(np_expected, np_actual), "Output should be unchanged"

    def test_callback_invoked(self):
        """Test that callback is invoked with correct arguments."""
        layer = nn.Linear(64, 64)

        calls = []
        def callback(layer_idx: int, hidden_state: mx.array):
            mx.eval(hidden_state)
            calls.append((layer_idx, tuple(hidden_state.shape)))

        verified = VerifiedLayer(layer, layer_idx=7, callback=callback)
        x = mx.random.normal((2, 16, 64))

        _ = verified(x)
        mx.eval(_)

        assert len(calls) == 1, "Callback should be invoked once"
        assert calls[0][0] == 7, "Layer index should be 7"
        assert calls[0][1] == (2, 16, 64), "Shape should match output"

    def test_attribute_delegation(self):
        """Test that attributes are delegated to original layer."""
        layer = nn.Linear(64, 128, bias=True)

        verified = VerifiedLayer(layer, layer_idx=0, callback=lambda *args: None)

        # Access attributes that should be delegated
        assert verified.weight.shape == (128, 64), "Weight shape should be accessible"

    def test_multiple_calls(self):
        """Test multiple forward passes through verified layer."""
        layer = nn.Linear(32, 32)

        call_count = [0]
        def callback(layer_idx: int, hidden_state: mx.array):
            call_count[0] += 1

        verified = VerifiedLayer(layer, layer_idx=0, callback=callback)

        for _ in range(5):
            x = mx.random.normal((1, 8, 32))
            _ = verified(x)
            mx.eval(_)

        assert call_count[0] == 5, "Callback should be called 5 times"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestWrapLayersForVerification:
    """Tests for wrap_layers_for_verification helper."""

    def test_wraps_all_layers(self):
        """Test that all layers are wrapped."""
        layers = [nn.Linear(32, 32) for _ in range(5)]

        calls = []
        def callback(layer_idx: int, hidden_state: mx.array):
            calls.append(layer_idx)

        wrapped = wrap_layers_for_verification(layers, start_layer=10, callback=callback)

        assert len(wrapped) == 5, "Should wrap all layers"

        # Execute each wrapped layer
        for w in wrapped:
            x = mx.random.normal((1, 4, 32))
            _ = w(x)
            mx.eval(_)

        # Check layer indices are correct
        assert sorted(calls) == [10, 11, 12, 13, 14], "Layer indices should be 10-14"

    def test_layer_idx_offset(self):
        """Test that start_layer offsets layer indices correctly."""
        layers = [nn.Linear(16, 16) for _ in range(3)]

        indices = []
        def callback(layer_idx: int, hidden_state: mx.array):
            indices.append(layer_idx)

        wrapped = wrap_layers_for_verification(layers, start_layer=22, callback=callback)

        for w in wrapped:
            x = mx.random.normal((1, 2, 16))
            _ = w(x)
            mx.eval(_)

        assert indices == [22, 23, 24], "Indices should be offset by start_layer"


class TestVerifiedLayerWithoutMLX:
    """Tests that work without MLX installed."""

    def test_module_imports(self):
        """Test that module can be imported without MLX."""
        # This test passes if we got this far without import errors
        from exo.consortium.exo_integration.verified_layer import (
            VerifiedLayer,
            wrap_layers_for_verification,
        )
        assert VerifiedLayer is not None
        assert wrap_layers_for_verification is not None

    def test_mock_layer(self):
        """Test VerifiedLayer with a mock layer (no MLX)."""
        # Create mock layer
        mock_layer = Mock()
        mock_output = Mock()
        mock_layer.return_value = mock_output

        # Create callback
        callback = Mock()

        # Create verified layer
        verified = VerifiedLayer(mock_layer, layer_idx=3, callback=callback)

        # Call it with mock input
        mock_input = Mock()
        result = verified(mock_input)

        # Verify
        mock_layer.assert_called_once_with(mock_input)
        callback.assert_called_once_with(3, mock_output)
        assert result is mock_output
