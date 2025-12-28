"""
KV Cache Management

This module handles Key-Value cache for transformer attention layers.
Each pipeline stage maintains its own KV cache for the layers it owns.

The KV cache is critical for:
1. Efficient autoregressive generation (reuse previous K/V)
2. Failover recovery (checkpoint and restore)
3. Correct attention computation across pipeline stages
"""

import gzip
import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayerKVCache:
    """KV cache for a single transformer layer."""

    key: np.ndarray  # [batch, num_heads, seq_len, head_dim]
    value: np.ndarray  # [batch, num_heads, seq_len, head_dim]

    @property
    def seq_len(self) -> int:
        """Current sequence length in cache."""
        return self.key.shape[2]

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self.key.shape[1]

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.key.shape[3]

    def append(self, new_key: np.ndarray, new_value: np.ndarray) -> None:
        """
        Append new K/V entries to the cache.

        Args:
            new_key: New key tensor [batch, num_heads, new_seq_len, head_dim]
            new_value: New value tensor [batch, num_heads, new_seq_len, head_dim]
        """
        self.key = np.concatenate([self.key, new_key], axis=2)
        self.value = np.concatenate([self.value, new_value], axis=2)

    def truncate(self, max_len: int) -> None:
        """Truncate cache to maximum length (for sliding window)."""
        if self.seq_len > max_len:
            self.key = self.key[:, :, -max_len:, :]
            self.value = self.value[:, :, -max_len:, :]


@dataclass
class StageKVCache:
    """
    KV cache for a pipeline stage (subset of layers).

    Manages KV cache for layers [layer_start, layer_end).
    """

    layer_start: int
    layer_end: int
    num_kv_heads: int = 8  # Llama-3-8B uses GQA with 8 KV heads
    head_dim: int = 128  # 4096 / 32 heads
    dtype: np.dtype = field(default_factory=lambda: np.float16)
    caches: Dict[int, LayerKVCache] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        """Number of layers in this stage."""
        return self.layer_end - self.layer_start

    @property
    def seq_len(self) -> int:
        """Current sequence length (from first layer's cache)."""
        if not self.caches:
            return 0
        first_cache = next(iter(self.caches.values()))
        return first_cache.seq_len

    def init_cache(
        self,
        batch_size: int = 1,
        initial_seq_len: int = 0,
    ) -> None:
        """
        Initialize empty KV caches for all layers.

        Args:
            batch_size: Batch size
            initial_seq_len: Initial sequence length (usually 0)
        """
        self.caches.clear()

        for layer_idx in range(self.layer_start, self.layer_end):
            local_idx = layer_idx - self.layer_start

            # Initialize with empty tensors
            shape = (batch_size, self.num_kv_heads, initial_seq_len, self.head_dim)

            self.caches[local_idx] = LayerKVCache(
                key=np.zeros(shape, dtype=self.dtype),
                value=np.zeros(shape, dtype=self.dtype),
            )

        logger.debug(
            f"Initialized KV cache for layers {self.layer_start}-{self.layer_end}"
        )

    def update_layer(
        self,
        layer_idx: int,
        new_key: np.ndarray,
        new_value: np.ndarray,
    ) -> None:
        """
        Update KV cache for a specific layer.

        Args:
            layer_idx: Global layer index
            new_key: New key tensor
            new_value: New value tensor
        """
        local_idx = layer_idx - self.layer_start

        if local_idx not in self.caches:
            # First update for this layer
            self.caches[local_idx] = LayerKVCache(key=new_key, value=new_value)
        else:
            self.caches[local_idx].append(new_key, new_value)

    def get_layer_kv(self, layer_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get KV cache for a layer.

        Args:
            layer_idx: Global layer index

        Returns:
            Tuple of (key, value) tensors or None if not cached
        """
        local_idx = layer_idx - self.layer_start

        if local_idx not in self.caches:
            return None

        cache = self.caches[local_idx]
        return cache.key, cache.value

    def get_past_kv_tuple(self) -> Tuple:
        """
        Get KV cache in HuggingFace transformers format.

        Returns:
            Tuple of (key, value) tuples for each layer
        """
        return tuple(
            (self.caches[i].key, self.caches[i].value)
            for i in range(self.num_layers)
            if i in self.caches
        )

    def clear(self) -> None:
        """Clear all cached values."""
        self.caches.clear()

    def serialize(self) -> bytes:
        """
        Serialize KV cache for checkpointing.

        Returns:
            Compressed bytes of the cache state
        """
        state = {
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "caches": {
                idx: {
                    "key": cache.key,
                    "value": cache.value,
                }
                for idx, cache in self.caches.items()
            },
        }

        # Compress for efficient storage/transfer
        return gzip.compress(pickle.dumps(state), compresslevel=6)

    @classmethod
    def deserialize(cls, data: bytes) -> "StageKVCache":
        """
        Restore KV cache from serialized data.

        Args:
            data: Compressed bytes from serialize()

        Returns:
            Restored StageKVCache instance
        """
        state = pickle.loads(gzip.decompress(data))

        cache = cls(
            layer_start=state["layer_start"],
            layer_end=state["layer_end"],
            num_kv_heads=state["num_kv_heads"],
            head_dim=state["head_dim"],
        )

        for idx, cache_data in state["caches"].items():
            cache.caches[idx] = LayerKVCache(
                key=cache_data["key"],
                value=cache_data["value"],
            )

        return cache

    def memory_usage_mb(self) -> float:
        """Calculate approximate memory usage in MB."""
        total_bytes = 0

        for layer_cache in self.caches.values():
            total_bytes += layer_cache.key.nbytes
            total_bytes += layer_cache.value.nbytes

        return total_bytes / (1024 * 1024)


class KVCacheCheckpointer:
    """
    Manages KV cache checkpoints for failover recovery.

    Periodically saves cache state so that a backup node can
    resume from a recent checkpoint rather than recomputing
    from the beginning.
    """

    def __init__(
        self,
        checkpoint_interval: int = 64,
        max_checkpoints: int = 3,
    ):
        """
        Initialize checkpointer.

        Args:
            checkpoint_interval: Tokens between checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Tuple[int, bytes]] = []  # (seq_pos, data)

    def maybe_checkpoint(
        self,
        kv_cache: StageKVCache,
        current_pos: int,
    ) -> bool:
        """
        Create checkpoint if enough tokens have passed.

        Args:
            kv_cache: Current KV cache state
            current_pos: Current sequence position

        Returns:
            True if checkpoint was created
        """
        # Check if we should checkpoint
        last_pos = self.checkpoints[-1][0] if self.checkpoints else 0

        if current_pos - last_pos < self.checkpoint_interval:
            return False

        # Create checkpoint
        data = kv_cache.serialize()
        self.checkpoints.append((current_pos, data))

        # Trim old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)

        logger.info(f"Created KV cache checkpoint at position {current_pos}")
        return True

    def get_latest_checkpoint(self) -> Optional[Tuple[int, bytes]]:
        """
        Get the most recent checkpoint.

        Returns:
            Tuple of (position, data) or None if no checkpoints
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def restore_from_checkpoint(
        self,
        target_pos: int,
    ) -> Optional[Tuple[int, StageKVCache]]:
        """
        Find and restore the best checkpoint for a target position.

        Args:
            target_pos: Target sequence position

        Returns:
            Tuple of (checkpoint_pos, cache) or None
        """
        # Find checkpoint just before target
        best = None

        for pos, data in self.checkpoints:
            if pos <= target_pos:
                best = (pos, data)

        if best is None:
            return None

        pos, data = best
        cache = StageKVCache.deserialize(data)

        logger.info(f"Restored KV cache from checkpoint at position {pos}")
        return pos, cache

    def clear(self) -> None:
        """Clear all checkpoints."""
        self.checkpoints.clear()
