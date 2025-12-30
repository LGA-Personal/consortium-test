"""
Activation Tensor Serialization

This module provides efficient serialization and deserialization of
activation tensors for network transfer between pipeline stages.

Format:
- Magic bytes: 'CACT' (4 bytes)
- Version: uint16 (2 bytes)
- Dtype: uint8 (1 byte) - 0=float16, 1=float32
- Ndim: uint8 (1 byte)
- Shape: ndim x uint64 (8 bytes each)
- Data: raw tensor bytes (little-endian)
"""

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Magic bytes for activation format
MAGIC = b"CACT"  # Consortium ACTivation
VERSION = 1

# Dtype mapping
DTYPE_FLOAT16 = 0
DTYPE_FLOAT32 = 1

DTYPE_MAP = {
    DTYPE_FLOAT16: np.float16,
    DTYPE_FLOAT32: np.float32,
}

DTYPE_REVERSE_MAP = {
    np.dtype(np.float16): DTYPE_FLOAT16,
    np.dtype(np.float32): DTYPE_FLOAT32,
}


class SerializationError(Exception):
    """Raised when serialization or deserialization fails."""

    pass


@dataclass
class ActivationHeader:
    """Header metadata for serialized activation tensors."""

    version: int
    dtype: int  # 0=float16, 1=float32
    ndim: int
    shape: Tuple[int, ...]

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        # Magic + version + dtype + ndim
        header = struct.pack("<4sHBB", MAGIC, self.version, self.dtype, self.ndim)

        # Shape dimensions
        for dim in self.shape:
            header += struct.pack("<Q", dim)

        return header

    @classmethod
    def from_bytes(cls, data: bytes) -> Tuple["ActivationHeader", int]:
        """
        Deserialize header from bytes.

        Args:
            data: Raw bytes containing header

        Returns:
            Tuple of (header, offset) where offset is bytes consumed
        """
        if len(data) < 8:
            raise SerializationError(f"Header too short: {len(data)} bytes")

        magic, version, dtype, ndim = struct.unpack("<4sHBB", data[:8])

        if magic != MAGIC:
            raise SerializationError(f"Invalid magic bytes: {magic!r}, expected {MAGIC!r}")

        if version != VERSION:
            raise SerializationError(
                f"Unsupported version: {version}, expected {VERSION}"
            )

        if dtype not in DTYPE_MAP:
            raise SerializationError(f"Unknown dtype: {dtype}")

        offset = 8
        required_len = offset + (ndim * 8)

        if len(data) < required_len:
            raise SerializationError(
                f"Header incomplete: need {required_len} bytes, got {len(data)}"
            )

        shape = []
        for _ in range(ndim):
            (dim,) = struct.unpack("<Q", data[offset : offset + 8])
            shape.append(dim)
            offset += 8

        return cls(version=version, dtype=dtype, ndim=ndim, shape=tuple(shape)), offset

    @property
    def numpy_dtype(self) -> np.dtype:
        """Get numpy dtype for this header."""
        return DTYPE_MAP[self.dtype]


def serialize_activation(
    tensor: np.ndarray,
    use_float16: bool = True,
) -> bytes:
    """
    Serialize activation tensor for network transfer.

    Args:
        tensor: Numpy array to serialize
        use_float16: If True, convert to float16 for smaller transfer size

    Returns:
        Serialized bytes
    """
    # Convert dtype if requested
    if use_float16:
        tensor = tensor.astype(np.float16)

    # Ensure contiguous memory layout
    tensor = np.ascontiguousarray(tensor)

    # Determine dtype code
    dtype_code = DTYPE_REVERSE_MAP.get(tensor.dtype)
    if dtype_code is None:
        raise SerializationError(
            f"Unsupported dtype: {tensor.dtype}. "
            f"Supported: {list(DTYPE_REVERSE_MAP.keys())}"
        )

    # Build header
    header = ActivationHeader(
        version=VERSION,
        dtype=dtype_code,
        ndim=len(tensor.shape),
        shape=tensor.shape,
    )

    # Serialize
    return header.to_bytes() + tensor.tobytes()


def deserialize_activation(data: bytes) -> np.ndarray:
    """
    Deserialize activation tensor from bytes.

    Args:
        data: Serialized bytes from serialize_activation

    Returns:
        Numpy array (as a writable copy)
    """
    # Parse header
    header, offset = ActivationHeader.from_bytes(data)

    # Calculate expected data size
    expected_size = np.prod(header.shape) * np.dtype(header.numpy_dtype).itemsize
    actual_size = len(data) - offset

    if actual_size != expected_size:
        raise SerializationError(
            f"Data size mismatch: expected {expected_size} bytes, got {actual_size}"
        )

    # Parse tensor data
    tensor = np.frombuffer(data[offset:], dtype=header.numpy_dtype)
    tensor = tensor.reshape(header.shape)

    # Return writable copy
    return tensor.copy()


def get_activation_size(shape: Tuple[int, ...], dtype: np.dtype = np.float16) -> int:
    """
    Calculate serialized size for an activation tensor.

    Args:
        shape: Tensor shape
        dtype: Data type (default float16)

    Returns:
        Total bytes when serialized
    """
    header_size = 8 + (len(shape) * 8)  # Magic + version + dtype + ndim + shape
    data_size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    return header_size + data_size


def estimate_transfer_size_mb(
    hidden_dim: int,
    seq_len: int,
    batch_size: int = 1,
) -> float:
    """
    Estimate activation transfer size in MB.

    For Llama-3-8B: hidden_dim=4096

    Args:
        hidden_dim: Model hidden dimension
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        Size in megabytes
    """
    shape = (batch_size, seq_len, hidden_dim)
    size_bytes = get_activation_size(shape)
    return size_bytes / (1024 * 1024)
