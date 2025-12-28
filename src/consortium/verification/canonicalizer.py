"""
Canonical-Grid Commitment Implementation

This module implements the fuzzy verification scheme that allows
heterogeneous hardware (CUDA, Metal) to produce matching commitment
hashes despite floating-point variance.

The canonicalization function:
1. Casts to float16
2. Snaps to a grid (round(x * grid_factor) / grid_factor)
3. Clamps to a fixed range
4. Serializes as little-endian bytes
5. Hashes with SHA-256

This produces a commitment that is deterministic within the tolerance
defined by the grid factor.
"""

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Default parameters from architecture spec
DEFAULT_GRID_FACTOR = 64  # ~0.0156 precision
DEFAULT_CLAMP_MIN = -100.0
DEFAULT_CLAMP_MAX = 100.0


@dataclass(frozen=True)
class CanonicalConfig:
    """Configuration for canonicalization parameters."""

    grid_factor: int = DEFAULT_GRID_FACTOR
    clamp_min: float = DEFAULT_CLAMP_MIN
    clamp_max: float = DEFAULT_CLAMP_MAX

    def __post_init__(self) -> None:
        if self.grid_factor <= 0:
            raise ValueError(f"grid_factor must be positive, got {self.grid_factor}")
        if self.clamp_min >= self.clamp_max:
            raise ValueError(
                f"clamp_min ({self.clamp_min}) must be less than clamp_max ({self.clamp_max})"
            )


# Default configuration instance
DEFAULT_CONFIG = CanonicalConfig()


def canonicalize(
    tensor: np.ndarray,
    config: Optional[CanonicalConfig] = None,
) -> np.ndarray:
    """
    Apply canonical grid transformation to a tensor.

    This function does NOT hash - it returns the canonicalized tensor
    for inspection or further processing.

    Args:
        tensor: Input activation tensor of any shape (float32 or float16)
        config: Canonicalization parameters (uses defaults if None)

    Returns:
        Canonicalized tensor as float16
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Step 1: Cast to float16
    y = tensor.astype(np.float16)

    # Step 1b: Handle inf and nan before grid snap
    # Replace nan with 0, +inf with clamp_max, -inf with clamp_min
    y = np.where(np.isnan(y), 0.0, y)
    y = np.where(np.isposinf(y), config.clamp_max, y)
    y = np.where(np.isneginf(y), config.clamp_min, y)

    # Step 2: Grid snap
    # y_grid = round(y * grid_factor) / grid_factor
    y_grid = np.round(y * float(config.grid_factor)) / float(config.grid_factor)

    # Step 3: Clamp to range
    y_clamped = np.clip(y_grid, config.clamp_min, config.clamp_max)

    # Ensure contiguous memory layout in C order
    y_contiguous = np.ascontiguousarray(y_clamped, dtype=np.float16)

    return y_contiguous


def compute_commitment(
    tensor: np.ndarray,
    config: Optional[CanonicalConfig] = None,
) -> bytes:
    """
    Compute the canonical-grid commitment hash for a tensor.

    This is the primary function for generating commitments during
    work execution and verification.

    Args:
        tensor: Input activation tensor of any shape
        config: Canonicalization parameters (uses defaults if None)

    Returns:
        32-byte SHA-256 hash (commitment)
    """
    # Canonicalize
    canonical = canonicalize(tensor, config)

    # Serialize as little-endian bytes
    # numpy float16 is already in native byte order, but we ensure
    # little-endian by checking and converting if necessary
    if canonical.dtype.byteorder == ">":
        canonical = canonical.byteswap().newbyteorder()

    serialized = canonical.tobytes()

    # SHA-256 hash
    commitment = hashlib.sha256(serialized).digest()

    return commitment


def canonicalize_and_hash(
    tensor: np.ndarray,
    config: Optional[CanonicalConfig] = None,
) -> bytes:
    """
    Alias for compute_commitment for compatibility with plan pseudocode.

    Args:
        tensor: Input activation tensor
        config: Canonicalization parameters

    Returns:
        32-byte SHA-256 commitment hash
    """
    return compute_commitment(tensor, config)


def verify_commitment(
    tensor: np.ndarray,
    expected_hash: bytes,
    config: Optional[CanonicalConfig] = None,
) -> bool:
    """
    Verify that a tensor matches an expected commitment.

    Args:
        tensor: Recomputed activation tensor
        expected_hash: 32-byte commitment from original computation
        config: Canonicalization parameters

    Returns:
        True if commitment matches, False otherwise
    """
    actual_hash = compute_commitment(tensor, config)
    return actual_hash == expected_hash


def commitment_to_hex(commitment: bytes) -> str:
    """Convert commitment bytes to hex string for display/logging."""
    return commitment.hex()


def hex_to_commitment(hex_str: str) -> bytes:
    """Convert hex string back to commitment bytes."""
    return bytes.fromhex(hex_str)


def compute_input_hash(data: bytes) -> bytes:
    """
    Compute hash of input data for receipt chaining.

    This is used to hash the serialized input activation in receipts,
    enabling verification of the computation chain.

    Args:
        data: Serialized input (e.g., from serialize_activation)

    Returns:
        32-byte SHA-256 hash
    """
    return hashlib.sha256(data).digest()


class CommitmentMismatchError(Exception):
    """Raised when a commitment verification fails."""

    def __init__(
        self,
        expected: bytes,
        actual: bytes,
        message: str = "Commitment mismatch",
    ):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{message}: expected {commitment_to_hex(expected)[:16]}..., "
            f"got {commitment_to_hex(actual)[:16]}..."
        )
