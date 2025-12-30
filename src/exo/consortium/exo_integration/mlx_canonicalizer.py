"""
MLX-optimized Canonicalizer

Provides efficient canonicalization for MLX arrays while maintaining
exact compatibility with the numpy reference implementation in
consortium.verification.canonicalizer.

The canonicalization process:
1. Cast to float16
2. Handle inf/nan values
3. Snap to grid: round(x * grid_factor) / grid_factor
4. Clamp to range
5. SHA-256 hash

This produces commitments that match the numpy version exactly,
enabling verification across heterogeneous hardware.
"""

import hashlib
from typing import TYPE_CHECKING, Optional

import numpy as np

# Import canonical config from existing module
from exo.consortium.verification.canonicalizer import (
    CanonicalConfig,
    DEFAULT_CONFIG,
    DEFAULT_GRID_FACTOR,
    DEFAULT_CLAMP_MIN,
    DEFAULT_CLAMP_MAX,
)

# Type hints for MLX (may not be installed on all systems)
if TYPE_CHECKING:
    import mlx.core as mx

# Runtime imports are conditional
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def mlx_compute_commitment(
    tensor: "mx.array",
    config: Optional[CanonicalConfig] = None,
) -> bytes:
    """
    Compute canonical-grid commitment from MLX array.

    This function produces commitments that exactly match the numpy
    reference implementation, enabling cross-platform verification.

    Args:
        tensor: MLX array (hidden state from transformer layer)
        config: Canonicalization parameters (uses defaults if None)

    Returns:
        32-byte SHA-256 commitment hash

    Note:
        Requires mx.eval() on the tensor before conversion to ensure
        the async computation graph is evaluated. This function calls
        mx.eval() internally for safety.

    Example:
        ```python
        hidden_state = layer(x)  # MLX array
        commitment = mlx_compute_commitment(hidden_state)
        # commitment is 32 bytes, can be compared with numpy version
        ```
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX is not available. Install with: pip install mlx")

    if config is None:
        config = DEFAULT_CONFIG

    # Ensure tensor is evaluated before numpy conversion
    # This prevents race conditions with MLX's async evaluation
    mx.eval(tensor)

    # Convert to numpy (unavoidable for SHA-256)
    # Using copy=False for efficiency when possible
    np_tensor = np.array(tensor, copy=False)

    # Step 1: Cast to float16
    y = np_tensor.astype(np.float16)

    # Step 2: Handle inf/nan values
    # Replace nan with 0, +inf with clamp_max, -inf with clamp_min
    y = np.where(np.isnan(y), 0.0, y)
    y = np.where(np.isposinf(y), config.clamp_max, y)
    y = np.where(np.isneginf(y), config.clamp_min, y)

    # Step 3: Grid snap
    # y_grid = round(y * grid_factor) / grid_factor
    y = np.round(y * float(config.grid_factor)) / float(config.grid_factor)

    # Step 4: Clamp to range
    y = np.clip(y, config.clamp_min, config.clamp_max)

    # Step 5: Ensure contiguous memory layout
    y = np.ascontiguousarray(y, dtype=np.float16)

    # Step 6: Handle byte order (ensure little-endian for consistency)
    if y.dtype.byteorder == ">":
        y = y.byteswap().newbyteorder()

    # Step 7: SHA-256 hash
    return hashlib.sha256(y.tobytes()).digest()


def mlx_canonicalize(
    tensor: "mx.array",
    config: Optional[CanonicalConfig] = None,
) -> "mx.array":
    """
    Apply canonicalization on MLX array, returning MLX array.

    Useful for debugging and inspection. The returned array
    has the canonical form that would be hashed.

    Args:
        tensor: MLX array to canonicalize
        config: Canonicalization parameters

    Returns:
        Canonicalized MLX array (float16)

    Note:
        This performs the canonicalization in MLX operations
        for efficiency, but the result should be equivalent
        to the numpy path.
    """
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX is not available. Install with: pip install mlx")

    if config is None:
        config = DEFAULT_CONFIG

    # Cast to float16
    y = tensor.astype(mx.float16)

    # MLX doesn't have isnan/isinf directly in the same way,
    # so we convert through numpy for this step
    mx.eval(y)
    np_y = np.array(y, copy=False)
    np_y = np.where(np.isnan(np_y), 0.0, np_y)
    np_y = np.where(np.isposinf(np_y), config.clamp_max, np_y)
    np_y = np.where(np.isneginf(np_y), config.clamp_min, np_y)
    y = mx.array(np_y.astype(np.float16))

    # Grid snap (MLX operations)
    y = mx.round(y * float(config.grid_factor)) / float(config.grid_factor)

    # Clamp
    y = mx.clip(y, config.clamp_min, config.clamp_max)

    return y


def verify_mlx_commitment(
    tensor: "mx.array",
    expected_hash: bytes,
    config: Optional[CanonicalConfig] = None,
) -> bool:
    """
    Verify that an MLX tensor matches an expected commitment.

    Args:
        tensor: Recomputed activation tensor (MLX array)
        expected_hash: 32-byte commitment from original computation
        config: Canonicalization parameters

    Returns:
        True if commitment matches, False otherwise
    """
    actual_hash = mlx_compute_commitment(tensor, config)
    return actual_hash == expected_hash


# Re-export config for convenience
__all__ = [
    "mlx_compute_commitment",
    "mlx_canonicalize",
    "verify_mlx_commitment",
    "CanonicalConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_GRID_FACTOR",
    "DEFAULT_CLAMP_MIN",
    "DEFAULT_CLAMP_MAX",
    "MLX_AVAILABLE",
]
