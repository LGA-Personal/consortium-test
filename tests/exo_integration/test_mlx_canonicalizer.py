"""
Tests for MLX canonicalizer.

Verifies that MLX canonicalization produces identical hashes
to the numpy reference implementation.
"""

import pytest
import numpy as np

from exo.consortium.verification.canonicalizer import (
    compute_commitment,
    canonicalize,
    CanonicalConfig,
    DEFAULT_CONFIG,
)

# Try to import MLX
try:
    import mlx.core as mx
    from exo.consortium.exo_integration.mlx_canonicalizer import (
        mlx_compute_commitment,
        mlx_canonicalize,
        MLX_AVAILABLE,
    )
    HAS_MLX = MLX_AVAILABLE
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMLXCanonicalizerMatchesNumpy:
    """Tests verifying MLX matches numpy implementation."""

    def test_simple_tensor(self):
        """Test with a simple tensor."""
        np_tensor = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        mx_tensor = mx.array(np_tensor)

        np_hash = compute_commitment(np_tensor)
        mlx_hash = mlx_compute_commitment(mx_tensor)

        assert np_hash == mlx_hash, "Hashes should match for simple tensor"

    def test_random_small_tensor(self):
        """Test with random small tensors."""
        np.random.seed(42)

        for _ in range(10):
            shape = (2, 16, 128)
            np_tensor = np.random.randn(*shape).astype(np.float32)
            mx_tensor = mx.array(np_tensor)

            np_hash = compute_commitment(np_tensor)
            mlx_hash = mlx_compute_commitment(mx_tensor)

            assert np_hash == mlx_hash, f"Hash mismatch for shape {shape}"

    def test_hidden_state_shape(self):
        """Test with typical hidden state shapes."""
        np.random.seed(123)

        # Typical shapes: [batch, seq_len, hidden_dim]
        shapes = [
            (1, 1, 4096),      # Single token decode
            (1, 128, 4096),    # Short sequence
            (2, 64, 4096),     # Batch of 2
        ]

        for shape in shapes:
            np_tensor = np.random.randn(*shape).astype(np.float32)
            mx_tensor = mx.array(np_tensor)

            np_hash = compute_commitment(np_tensor)
            mlx_hash = mlx_compute_commitment(mx_tensor)

            assert np_hash == mlx_hash, f"Hash mismatch for shape {shape}"

    def test_extreme_values(self):
        """Test with extreme values (inf, nan, large numbers)."""
        np_tensor = np.array([
            [np.inf, -np.inf, np.nan],
            [1e10, -1e10, 0.0],
            [1e-10, -1e-10, 1.0],
        ], dtype=np.float32)
        mx_tensor = mx.array(np_tensor)

        np_hash = compute_commitment(np_tensor)
        mlx_hash = mlx_compute_commitment(mx_tensor)

        assert np_hash == mlx_hash, "Hashes should match for extreme values"

    def test_custom_config(self):
        """Test with custom canonicalization config."""
        config = CanonicalConfig(
            grid_factor=32,
            clamp_min=-50.0,
            clamp_max=50.0,
        )

        np.random.seed(456)
        np_tensor = np.random.randn(2, 32, 256).astype(np.float32)
        mx_tensor = mx.array(np_tensor)

        np_hash = compute_commitment(np_tensor, config)
        mlx_hash = mlx_compute_commitment(mx_tensor, config)

        assert np_hash == mlx_hash, "Hashes should match with custom config"

    def test_large_batch_stress(self):
        """Stress test with many random tensors."""
        np.random.seed(789)
        mismatches = 0

        for i in range(100):
            shape = (2, 128, 4096)
            np_tensor = np.random.randn(*shape).astype(np.float32)
            mx_tensor = mx.array(np_tensor)

            np_hash = compute_commitment(np_tensor)
            mlx_hash = mlx_compute_commitment(mx_tensor)

            if np_hash != mlx_hash:
                mismatches += 1

        assert mismatches == 0, f"Found {mismatches}/100 mismatches in stress test"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMLXCanonicalize:
    """Tests for the canonicalize function."""

    def test_canonicalize_returns_mlx_array(self):
        """Test that mlx_canonicalize returns an MLX array."""
        np_tensor = np.random.randn(2, 16, 128).astype(np.float32)
        mx_tensor = mx.array(np_tensor)

        result = mlx_canonicalize(mx_tensor)

        assert isinstance(result, mx.array), "Should return mx.array"
        assert result.dtype == mx.float16, "Should be float16"

    def test_canonicalize_shape_preserved(self):
        """Test that shape is preserved after canonicalization."""
        shape = (4, 32, 256)
        np_tensor = np.random.randn(*shape).astype(np.float32)
        mx_tensor = mx.array(np_tensor)

        result = mlx_canonicalize(mx_tensor)

        assert result.shape == shape, "Shape should be preserved"


class TestNumpyCanonicalization:
    """Tests for numpy canonicalization (always runs)."""

    def test_deterministic(self):
        """Test that canonicalization is deterministic."""
        np.random.seed(111)
        tensor = np.random.randn(2, 16, 128).astype(np.float32)

        hash1 = compute_commitment(tensor)
        hash2 = compute_commitment(tensor)

        assert hash1 == hash2, "Same tensor should produce same hash"

    def test_different_tensors_different_hashes(self):
        """Test that different tensors produce different hashes."""
        np.random.seed(222)
        tensor1 = np.random.randn(2, 16, 128).astype(np.float32)
        tensor2 = np.random.randn(2, 16, 128).astype(np.float32)

        hash1 = compute_commitment(tensor1)
        hash2 = compute_commitment(tensor2)

        assert hash1 != hash2, "Different tensors should produce different hashes"

    def test_hash_length(self):
        """Test that hash is 32 bytes (SHA-256)."""
        tensor = np.random.randn(2, 16, 128).astype(np.float32)
        hash_val = compute_commitment(tensor)

        assert len(hash_val) == 32, "SHA-256 should produce 32 bytes"

    def test_grid_snap_effect(self):
        """Test that grid snap groups similar values."""
        config = CanonicalConfig(grid_factor=64)

        # Values that should snap to the same grid point
        tensor1 = np.array([[1.0]], dtype=np.float32)
        tensor2 = np.array([[1.007]], dtype=np.float32)  # Within 1/64

        canonical1 = canonicalize(tensor1, config)
        canonical2 = canonicalize(tensor2, config)

        # Both should snap to same value
        assert np.allclose(canonical1, canonical2), (
            "Similar values should snap to same grid point"
        )

    def test_clamp_effect(self):
        """Test that values are clamped to range."""
        config = CanonicalConfig(clamp_min=-100.0, clamp_max=100.0)

        tensor = np.array([[200.0, -200.0, 50.0]], dtype=np.float32)
        canonical = canonicalize(tensor, config)

        assert canonical[0, 0] == 100.0, "Should clamp to max"
        assert canonical[0, 1] == -100.0, "Should clamp to min"
        # Note: 50.0 gets grid-snapped, may not be exactly 50.0
