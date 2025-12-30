"""
Tests for activation tensor serialization.
"""

import numpy as np
import pytest

from exo.consortium.transport.serialization import (
    serialize_activation,
    deserialize_activation,
    get_activation_size,
    estimate_transfer_size_mb,
    SerializationError,
    MAGIC,
    VERSION,
)


class TestSerialization:
    """Tests for serialize/deserialize roundtrip."""

    def test_roundtrip_float32(self, sample_activation):
        """Test serialization roundtrip with float32 input."""
        serialized = serialize_activation(sample_activation, use_float16=False)
        deserialized = deserialize_activation(serialized)

        assert deserialized.dtype == np.float32
        assert deserialized.shape == sample_activation.shape
        np.testing.assert_allclose(deserialized, sample_activation)

    def test_roundtrip_float16(self, sample_activation):
        """Test serialization with conversion to float16."""
        serialized = serialize_activation(sample_activation, use_float16=True)
        deserialized = deserialize_activation(serialized)

        assert deserialized.dtype == np.float16
        assert deserialized.shape == sample_activation.shape
        # Less precision due to float16 conversion
        np.testing.assert_allclose(
            deserialized.astype(np.float32),
            sample_activation,
            rtol=1e-3,
        )

    def test_magic_bytes(self, sample_activation):
        """Serialized data should start with magic bytes."""
        serialized = serialize_activation(sample_activation)
        assert serialized[:4] == MAGIC

    def test_version_in_header(self, sample_activation):
        """Serialized data should contain version."""
        serialized = serialize_activation(sample_activation)
        # Version is bytes 4-5 (uint16 little-endian)
        version = int.from_bytes(serialized[4:6], "little")
        assert version == VERSION

    def test_output_is_writable(self, sample_activation):
        """Deserialized array should be writable."""
        serialized = serialize_activation(sample_activation)
        deserialized = deserialize_activation(serialized)

        # This should not raise
        deserialized[0, 0, 0] = 999.0
        assert deserialized[0, 0, 0] == pytest.approx(999.0, rel=1e-3)

    def test_different_shapes(self):
        """Test various tensor shapes."""
        shapes = [
            (1, 1, 1),
            (1, 64, 4096),
            (2, 10, 4096),
            (1, 1, 1024),
        ]

        for shape in shapes:
            tensor = np.random.randn(*shape).astype(np.float32)
            serialized = serialize_activation(tensor)
            deserialized = deserialize_activation(serialized)
            assert deserialized.shape == shape

    def test_non_contiguous_input(self, sample_activation):
        """Non-contiguous arrays should still serialize correctly."""
        # Create non-contiguous view
        non_contiguous = sample_activation[::2, ::2, ::2]
        assert not non_contiguous.flags["C_CONTIGUOUS"]

        serialized = serialize_activation(non_contiguous)
        deserialized = deserialize_activation(serialized)

        np.testing.assert_allclose(
            deserialized.astype(np.float32),
            non_contiguous,
            rtol=1e-3,
        )


class TestSerializationErrors:
    """Tests for error handling."""

    def test_invalid_magic(self):
        """Invalid magic bytes should raise error."""
        bad_data = b"XXXX" + b"\x00" * 100
        with pytest.raises(SerializationError, match="Invalid magic"):
            deserialize_activation(bad_data)

    def test_unsupported_version(self):
        """Unsupported version should raise error."""
        # Create data with wrong version
        bad_data = MAGIC + b"\xff\xff" + b"\x00\x01" + b"\x00" * 100
        with pytest.raises(SerializationError, match="Unsupported version"):
            deserialize_activation(bad_data)

    def test_truncated_header(self):
        """Truncated header should raise error."""
        bad_data = MAGIC + b"\x01"  # Too short
        with pytest.raises(SerializationError, match="Header too short"):
            deserialize_activation(bad_data)

    def test_truncated_data(self, sample_activation):
        """Truncated data should raise error."""
        serialized = serialize_activation(sample_activation)
        truncated = serialized[:-100]  # Remove some bytes
        with pytest.raises(SerializationError, match="Data size mismatch"):
            deserialize_activation(truncated)


class TestSizeEstimation:
    """Tests for size estimation utilities."""

    def test_get_activation_size(self):
        """Test size calculation."""
        # Shape (1, 64, 4096) with float16: 1 * 64 * 4096 * 2 = 524288 bytes data
        # Plus header: 8 + (3 * 8) = 32 bytes
        shape = (1, 64, 4096)
        size = get_activation_size(shape)

        expected_data = 1 * 64 * 4096 * 2
        expected_header = 8 + (3 * 8)
        assert size == expected_data + expected_header

    def test_estimate_transfer_size(self):
        """Test MB estimation for Llama-3-8B activations."""
        # Single token with hidden_dim=4096
        size_mb = estimate_transfer_size_mb(hidden_dim=4096, seq_len=1)
        # Should be small (< 0.01 MB for single token)
        assert size_mb < 0.01

        # 64 tokens
        size_mb = estimate_transfer_size_mb(hidden_dim=4096, seq_len=64)
        # Should be around 0.5 MB
        assert 0.4 < size_mb < 0.6
