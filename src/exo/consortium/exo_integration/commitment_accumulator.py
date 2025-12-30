"""
Commitment Accumulator

Thread-safe accumulator for per-layer commitments during token generation.
Collects commitments from VerifiedLayer callbacks and provides batch
retrieval for transmission to the consortium coordinator.

The accumulator is designed to work with MLX's async evaluation model
and supports both synchronous and asynchronous usage patterns.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Optional

from .mlx_canonicalizer import mlx_compute_commitment, MLX_AVAILABLE

if TYPE_CHECKING:
    import mlx.core as mx

# Type alias for commitment (32-byte SHA-256 hash)
Commitment = bytes


@dataclass
class CommitmentBatch:
    """
    Batch of commitments for a single token generation.

    Attributes:
        token_index: Index of the generated token (0-based)
        commitments: Mapping from layer_idx to 32-byte commitment
        device_rank: Rank of the device that generated these commitments
        timestamp_ms: Unix timestamp in milliseconds when batch was created
    """
    token_index: int
    commitments: Dict[int, Commitment]
    device_rank: int
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def __len__(self) -> int:
        """Number of commitments in this batch."""
        return len(self.commitments)

    def __contains__(self, layer_idx: int) -> bool:
        """Check if a layer's commitment is in this batch."""
        return layer_idx in self.commitments


class CommitmentAccumulator:
    """
    Accumulates layer commitments during token generation.

    Thread-safe for use with MLX's async evaluation. Commitments are
    computed and stored synchronously, then flushed as a batch after
    each token generation completes.

    Usage:
        ```python
        accumulator = CommitmentAccumulator(device_rank=0)

        # During generation, VerifiedLayer calls this:
        accumulator.record(layer_idx=5, hidden_state=output)

        # After token is generated:
        if accumulator.has_commitments():
            batch = accumulator.flush()
            send_to_coordinator(batch)
        ```
    """

    def __init__(self, device_rank: int = 0):
        """
        Initialize the accumulator.

        Args:
            device_rank: Rank of this device in the distributed group
        """
        self._lock = threading.Lock()
        self._commitments: Dict[int, Commitment] = {}
        self._token_index: int = 0
        self._device_rank: int = device_rank

    def record(self, layer_idx: int, hidden_state: "mx.array") -> None:
        """
        Record commitment for a layer's hidden state.

        Computes the canonical-grid commitment and stores it.
        This method is called by VerifiedLayer after each forward pass.

        Args:
            layer_idx: Global layer index (0 to n_layers-1)
            hidden_state: Output hidden state tensor from the layer

        Note:
            This method calls mx.eval() internally to ensure the tensor
            is evaluated before computing the commitment.
        """
        # Compute commitment (handles mx.eval internally)
        commitment = mlx_compute_commitment(hidden_state)

        # TEMPORARY LOGGING: Show commitment hash
        print(f"   ✅ Commitment: {commitment[:8].hex()}... (layer {layer_idx})")

        with self._lock:
            self._commitments[layer_idx] = commitment

    def record_precomputed(self, layer_idx: int, commitment: Commitment) -> None:
        """
        Record a pre-computed commitment.

        Useful when the commitment has already been computed externally
        or when testing with mock commitments.

        Args:
            layer_idx: Global layer index
            commitment: 32-byte SHA-256 hash
        """
        if len(commitment) != 32:
            raise ValueError(f"Commitment must be 32 bytes, got {len(commitment)}")

        with self._lock:
            self._commitments[layer_idx] = commitment

    def has_commitments(self) -> bool:
        """Check if any commitments have been recorded."""
        with self._lock:
            return len(self._commitments) > 0

    def flush(self) -> CommitmentBatch:
        """
        Flush and return accumulated commitments as a batch.

        Clears the accumulator and increments the token index.

        Returns:
            CommitmentBatch containing all accumulated commitments

        Raises:
            ValueError: If no commitments have been recorded
        """
        with self._lock:
            if not self._commitments:
                raise ValueError("No commitments to flush")

            batch = CommitmentBatch(
                token_index=self._token_index,
                commitments=self._commitments.copy(),
                device_rank=self._device_rank,
            )

            self._commitments.clear()
            self._token_index += 1

            return batch

    def peek(self) -> Dict[int, Commitment]:
        """
        Get current commitments without flushing.

        Returns:
            Copy of current commitments dict
        """
        with self._lock:
            return self._commitments.copy()

    def set_token_index(self, idx: int) -> None:
        """
        Set current token index for tracking.

        Useful when resuming generation or synchronizing with
        external state.

        Args:
            idx: Token index to set
        """
        with self._lock:
            self._token_index = idx

    def get_token_index(self) -> int:
        """Get current token index."""
        with self._lock:
            return self._token_index

    def clear(self) -> None:
        """Clear accumulated commitments without flushing."""
        with self._lock:
            self._commitments.clear()

    def reset(self) -> None:
        """Reset accumulator to initial state (clear and reset token index)."""
        with self._lock:
            self._commitments.clear()
            self._token_index = 0

    @property
    def device_rank(self) -> int:
        """Get the device rank."""
        return self._device_rank

    @device_rank.setter
    def device_rank(self, value: int) -> None:
        """Set the device rank."""
        with self._lock:
            self._device_rank = value

    def __len__(self) -> int:
        """Number of currently accumulated commitments."""
        with self._lock:
            return len(self._commitments)

    def create_callback(self) -> Callable[[int, "mx.array"], None]:
        """
        Create a callback function for use with VerifiedLayer.

        Returns a callback that, when called, records commitments
        in this accumulator.

        Returns:
            Callback function matching VerificationCallback protocol

        Example:
            ```python
            accumulator = CommitmentAccumulator()
            callback = accumulator.create_callback()
            verified_layer = VerifiedLayer(layer, layer_idx=0, callback=callback)
            ```
        """
        def callback(layer_idx: int, hidden_state: "mx.array") -> None:
            self.record(layer_idx, hidden_state)

        return callback


class MultiDeviceAccumulator:
    """
    Manages commitment accumulators for multiple devices.

    Useful for coordinating verification across a distributed pipeline
    where each device has its own accumulator.
    """

    def __init__(self):
        self._accumulators: Dict[int, CommitmentAccumulator] = {}
        self._lock = threading.Lock()

    def get_or_create(self, device_rank: int) -> CommitmentAccumulator:
        """
        Get or create accumulator for a device.

        Args:
            device_rank: Device rank

        Returns:
            CommitmentAccumulator for the device
        """
        with self._lock:
            if device_rank not in self._accumulators:
                self._accumulators[device_rank] = CommitmentAccumulator(device_rank)
            return self._accumulators[device_rank]

    def flush_all(self) -> Dict[int, CommitmentBatch]:
        """
        Flush all accumulators that have commitments.

        Returns:
            Dict mapping device_rank to CommitmentBatch
        """
        batches: Dict[int, CommitmentBatch] = {}

        with self._lock:
            for device_rank, accumulator in self._accumulators.items():
                if accumulator.has_commitments():
                    batches[device_rank] = accumulator.flush()

        return batches

    def reset_all(self) -> None:
        """Reset all accumulators."""
        with self._lock:
            for accumulator in self._accumulators.values():
                accumulator.reset()

    def __getitem__(self, device_rank: int) -> CommitmentAccumulator:
        """Get accumulator by device rank."""
        return self.get_or_create(device_rank)
