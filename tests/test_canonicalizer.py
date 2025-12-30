"""
Tests for the canonical-grid commitment system.

These tests validate that:
1. Canonicalization is deterministic
2. Grid snapping works correctly
3. Clamping handles extreme values
4. Commitments are stable across runs
5. Verification works correctly
"""

import numpy as np
import pytest

from exo.consortium.verification.canonicalizer import (
    CanonicalConfig,
    canonicalize,
    canonicalize_and_hash,
    commitment_to_hex,
    compute_commitment,
    verify_commitment,
    CommitmentMismatchError,
    DEFAULT_CONFIG,
)


class TestCanonicalization:
    """Tests for the canonicalize function."""

    def test_output_dtype_is_float16(self, sample_activation):
        """Canonicalized output should always be float16."""
        result = canonicalize(sample_activation)
        assert result.dtype == np.float16

    def test_output_shape_preserved(self, sample_activation):
        """Shape should be preserved."""
        result = canonicalize(sample_activation)
        assert result.shape == sample_activation.shape

    def test_grid_snapping(self):
        """Test that values snap to grid correctly."""
        # With grid_factor=64, values should snap to multiples of 1/64 = 0.015625
        tensor = np.array([0.1, 0.5, 1.0, -0.3], dtype=np.float32)
        result = canonicalize(tensor)

        # Check that values are on the grid
        grid_factor = DEFAULT_CONFIG.grid_factor
        for val in result:
            # Value * grid_factor should be close to an integer
            scaled = float(val) * grid_factor
            assert abs(scaled - round(scaled)) < 1e-3, f"Value {val} not on grid"

    def test_clamping(self, sample_activation_with_extremes):
        """Test that extreme values are clamped."""
        result = canonicalize(sample_activation_with_extremes)

        # All values should be within clamp range
        assert np.all(result >= DEFAULT_CONFIG.clamp_min)
        assert np.all(result <= DEFAULT_CONFIG.clamp_max)

        # inf/nan should be handled (clamped or replaced)
        assert np.all(np.isfinite(result))

    def test_determinism(self, sample_activation):
        """Same input should produce same output."""
        result1 = canonicalize(sample_activation)
        result2 = canonicalize(sample_activation)
        assert np.array_equal(result1, result2)

    def test_custom_config(self, sample_activation):
        """Custom config should be applied."""
        config = CanonicalConfig(grid_factor=128, clamp_min=-50.0, clamp_max=50.0)
        result = canonicalize(sample_activation, config)

        # Check clamping with custom range
        assert np.all(result >= -50.0)
        assert np.all(result <= 50.0)


class TestCommitment:
    """Tests for commitment generation."""

    def test_commitment_is_32_bytes(self, sample_activation):
        """Commitment should be exactly 32 bytes (SHA-256)."""
        commitment = compute_commitment(sample_activation)
        assert len(commitment) == 32

    def test_commitment_determinism(self, sample_activation):
        """Same input should produce same commitment."""
        c1 = compute_commitment(sample_activation)
        c2 = compute_commitment(sample_activation)
        assert c1 == c2

    def test_different_inputs_different_commitments(self, sample_activation):
        """Different inputs should (almost certainly) produce different commitments."""
        c1 = compute_commitment(sample_activation)

        # Modify one value
        modified = sample_activation.copy()
        modified[0, 0, 0] += 1.0
        c2 = compute_commitment(modified)

        assert c1 != c2

    def test_commitment_hex_conversion(self, sample_activation):
        """Test hex string conversion."""
        commitment = compute_commitment(sample_activation)
        hex_str = commitment_to_hex(commitment)

        assert len(hex_str) == 64  # 32 bytes * 2 hex chars
        assert all(c in "0123456789abcdef" for c in hex_str)

    def test_canonicalize_and_hash_alias(self, sample_activation):
        """canonicalize_and_hash should be identical to compute_commitment."""
        c1 = compute_commitment(sample_activation)
        c2 = canonicalize_and_hash(sample_activation)
        assert c1 == c2


class TestVerification:
    """Tests for commitment verification."""

    def test_verify_correct_commitment(self, sample_activation):
        """Verification should pass for matching commitment."""
        commitment = compute_commitment(sample_activation)
        assert verify_commitment(sample_activation, commitment) is True

    def test_verify_wrong_commitment(self, sample_activation):
        """Verification should fail for wrong commitment."""
        commitment = compute_commitment(sample_activation)
        wrong_commitment = b"\x00" * 32
        assert verify_commitment(sample_activation, wrong_commitment) is False

    def test_verify_modified_tensor(self, sample_activation):
        """Verification should fail if tensor is modified."""
        commitment = compute_commitment(sample_activation)

        modified = sample_activation.copy()
        modified[0, 0, 0] += 1.0

        assert verify_commitment(modified, commitment) is False

    def test_verify_small_change_within_grid(self, sample_activation):
        """Very small changes within grid tolerance should still match."""
        commitment = compute_commitment(sample_activation)

        # Make a change smaller than grid resolution (1/64 = 0.015625)
        modified = sample_activation.copy()
        modified[0, 0, 0] += 0.001  # Much smaller than grid resolution

        # This should still match because both snap to the same grid point
        assert verify_commitment(modified, commitment) is True


class TestCrossHardwareSimulation:
    """
    Tests simulating cross-hardware verification.

    In practice, CUDA and Metal may produce slightly different
    floating-point results. The canonical grid should absorb these
    differences.
    """

    def test_simulated_hardware_variance(self):
        """
        Test that small perturbations well within grid cells produce same hash.

        The grid factor of 64 means each cell is 1/64 = 0.015625 wide.
        Perturbations much smaller than half the grid width should be absorbed.
        """
        # Create a simple tensor with values clearly centered in grid cells
        # Use multiples of 1/64 to be exactly on grid, then verify small perturbations
        grid_factor = 64

        # Values that are exactly on grid points
        base_values = np.array([[[0.0, 0.5, 1.0, -0.5, -1.0, 2.0]]], dtype=np.float32)

        # Compute commitment
        commitment = compute_commitment(base_values)

        # Add very small perturbation (much less than half grid width = 0.0078125)
        perturbation = 0.001  # This is 0.001, well under 0.0078
        perturbed = base_values + perturbation

        # Both should produce the same commitment because perturbation is
        # small enough that all values still round to the same grid points
        assert verify_commitment(perturbed, commitment) is True

    def test_grid_boundary_sensitivity(self):
        """Test that values near grid boundaries can flip with small changes."""
        # Create a value exactly on a grid boundary (0.5/64 from a grid point)
        grid_factor = 64
        grid_point = 1.0
        boundary_value = grid_point + 0.5 / grid_factor  # Exactly halfway

        tensor1 = np.array([[[boundary_value - 0.001]]], dtype=np.float32)
        tensor2 = np.array([[[boundary_value + 0.001]]], dtype=np.float32)

        # These should produce different commitments (they round to different grid points)
        c1 = compute_commitment(tensor1)
        c2 = compute_commitment(tensor2)
        assert c1 != c2, "Values on opposite sides of grid boundary should differ"

    def test_larger_variance_fails(self, sample_activation):
        """Larger variance should cause verification failure."""
        commitment = compute_commitment(sample_activation)

        # Add larger noise that exceeds grid tolerance
        np.random.seed(456)
        noise = np.random.uniform(-0.1, 0.1, sample_activation.shape)
        noisy = sample_activation + noise

        # This should fail
        assert verify_commitment(noisy.astype(np.float32), commitment) is False


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_invalid_grid_factor(self):
        """Grid factor must be positive."""
        with pytest.raises(ValueError):
            CanonicalConfig(grid_factor=0)

        with pytest.raises(ValueError):
            CanonicalConfig(grid_factor=-1)

    def test_invalid_clamp_range(self):
        """Clamp min must be less than max."""
        with pytest.raises(ValueError):
            CanonicalConfig(clamp_min=100.0, clamp_max=-100.0)

        with pytest.raises(ValueError):
            CanonicalConfig(clamp_min=0.0, clamp_max=0.0)
