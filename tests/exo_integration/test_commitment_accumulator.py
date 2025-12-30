"""
Tests for CommitmentAccumulator.

Tests thread safety, flush behavior, and callback integration.
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch
import numpy as np

# Try to import MLX
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from exo.consortium.exo_integration.commitment_accumulator import (
    CommitmentAccumulator,
    CommitmentBatch,
    MultiDeviceAccumulator,
)


class TestCommitmentBatch:
    """Tests for CommitmentBatch dataclass."""

    def test_creation(self):
        """Test creating a commitment batch."""
        commitments = {0: b"x" * 32, 1: b"y" * 32}
        batch = CommitmentBatch(
            token_index=5,
            commitments=commitments,
            device_rank=1,
        )

        assert batch.token_index == 5
        assert batch.device_rank == 1
        assert len(batch) == 2
        assert 0 in batch
        assert 2 not in batch

    def test_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        batch = CommitmentBatch(
            token_index=0,
            commitments={},
            device_rank=0,
        )

        assert batch.timestamp_ms > 0
        # Should be recent (within last minute)
        assert batch.timestamp_ms > int(time.time() * 1000) - 60000


class TestCommitmentAccumulator:
    """Tests for CommitmentAccumulator."""

    def test_empty_initially(self):
        """Test accumulator starts empty."""
        acc = CommitmentAccumulator()

        assert not acc.has_commitments()
        assert len(acc) == 0
        assert acc.get_token_index() == 0

    def test_record_precomputed(self):
        """Test recording pre-computed commitments."""
        acc = CommitmentAccumulator()

        commitment1 = b"a" * 32
        commitment2 = b"b" * 32

        acc.record_precomputed(0, commitment1)
        acc.record_precomputed(5, commitment2)

        assert acc.has_commitments()
        assert len(acc) == 2

        peek = acc.peek()
        assert peek[0] == commitment1
        assert peek[5] == commitment2

    def test_record_precomputed_validates_length(self):
        """Test that invalid commitment length raises error."""
        acc = CommitmentAccumulator()

        with pytest.raises(ValueError, match="32 bytes"):
            acc.record_precomputed(0, b"too short")

    def test_flush_returns_batch(self):
        """Test flush returns correct batch."""
        acc = CommitmentAccumulator(device_rank=2)

        acc.record_precomputed(0, b"x" * 32)
        acc.record_precomputed(1, b"y" * 32)

        batch = acc.flush()

        assert isinstance(batch, CommitmentBatch)
        assert batch.token_index == 0
        assert batch.device_rank == 2
        assert len(batch.commitments) == 2
        assert batch.commitments[0] == b"x" * 32

    def test_flush_increments_token_index(self):
        """Test flush increments token index."""
        acc = CommitmentAccumulator()

        acc.record_precomputed(0, b"x" * 32)
        acc.flush()
        assert acc.get_token_index() == 1

        acc.record_precomputed(0, b"y" * 32)
        batch2 = acc.flush()
        assert batch2.token_index == 1
        assert acc.get_token_index() == 2

    def test_flush_clears_commitments(self):
        """Test flush clears accumulated commitments."""
        acc = CommitmentAccumulator()

        acc.record_precomputed(0, b"x" * 32)
        acc.flush()

        assert not acc.has_commitments()
        assert len(acc) == 0

    def test_flush_empty_raises_error(self):
        """Test flushing empty accumulator raises error."""
        acc = CommitmentAccumulator()

        with pytest.raises(ValueError, match="No commitments"):
            acc.flush()

    def test_clear(self):
        """Test clear removes commitments without incrementing token."""
        acc = CommitmentAccumulator()

        acc.record_precomputed(0, b"x" * 32)
        acc.clear()

        assert not acc.has_commitments()
        assert acc.get_token_index() == 0  # Not incremented

    def test_reset(self):
        """Test reset clears everything."""
        acc = CommitmentAccumulator()

        acc.record_precomputed(0, b"x" * 32)
        acc.flush()
        acc.record_precomputed(0, b"y" * 32)
        acc.reset()

        assert not acc.has_commitments()
        assert acc.get_token_index() == 0

    def test_set_token_index(self):
        """Test manually setting token index."""
        acc = CommitmentAccumulator()

        acc.set_token_index(100)
        assert acc.get_token_index() == 100

        acc.record_precomputed(0, b"x" * 32)
        batch = acc.flush()
        assert batch.token_index == 100

    def test_device_rank_property(self):
        """Test device_rank property."""
        acc = CommitmentAccumulator(device_rank=5)

        assert acc.device_rank == 5

        acc.device_rank = 10
        assert acc.device_rank == 10

    def test_create_callback(self):
        """Test create_callback returns working callback."""
        acc = CommitmentAccumulator()

        callback = acc.create_callback()
        assert callable(callback)

        # Can't test without MLX, but verify it returns callable

    def test_thread_safety(self):
        """Test accumulator is thread-safe."""
        acc = CommitmentAccumulator()
        errors = []

        def record_thread(thread_id: int, count: int):
            try:
                for i in range(count):
                    commitment = bytes([thread_id] * 32)
                    acc.record_precomputed(thread_id * 1000 + i, commitment)
            except Exception as e:
                errors.append(e)

        threads = []
        for t in range(5):
            thread = threading.Thread(target=record_thread, args=(t, 20))
            threads.append(thread)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(acc) == 100  # 5 threads * 20 commitments each


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestCommitmentAccumulatorWithMLX:
    """Tests that require MLX."""

    def test_record_mlx_tensor(self):
        """Test recording commitment from MLX tensor."""
        acc = CommitmentAccumulator()

        # Create a simple MLX tensor
        tensor = mx.array(np.random.randn(2, 4, 16).astype(np.float32))

        acc.record(layer_idx=5, hidden_state=tensor)

        assert acc.has_commitments()
        peek = acc.peek()
        assert 5 in peek
        assert len(peek[5]) == 32  # SHA-256

    def test_callback_integration(self):
        """Test callback from create_callback works with MLX."""
        acc = CommitmentAccumulator()
        callback = acc.create_callback()

        tensor = mx.array(np.random.randn(1, 2, 8).astype(np.float32))
        callback(3, tensor)

        assert len(acc) == 1
        assert 3 in acc.peek()


class TestMultiDeviceAccumulator:
    """Tests for MultiDeviceAccumulator."""

    def test_get_or_create(self):
        """Test getting/creating accumulators."""
        multi = MultiDeviceAccumulator()

        acc0 = multi.get_or_create(0)
        acc1 = multi.get_or_create(1)

        assert isinstance(acc0, CommitmentAccumulator)
        assert acc0.device_rank == 0
        assert acc1.device_rank == 1

        # Same accumulator returned on second call
        assert multi.get_or_create(0) is acc0

    def test_getitem(self):
        """Test [] access."""
        multi = MultiDeviceAccumulator()

        acc = multi[5]
        assert acc.device_rank == 5

    def test_flush_all(self):
        """Test flushing all accumulators."""
        multi = MultiDeviceAccumulator()

        # Add commitments to multiple devices
        multi[0].record_precomputed(0, b"a" * 32)
        multi[1].record_precomputed(0, b"b" * 32)
        multi[2]  # Create but don't add commitments

        batches = multi.flush_all()

        assert len(batches) == 2  # Only 2 had commitments
        assert 0 in batches
        assert 1 in batches
        assert 2 not in batches

    def test_reset_all(self):
        """Test resetting all accumulators."""
        multi = MultiDeviceAccumulator()

        multi[0].record_precomputed(0, b"a" * 32)
        multi[0].flush()
        multi[1].record_precomputed(0, b"b" * 32)

        multi.reset_all()

        assert not multi[0].has_commitments()
        assert not multi[1].has_commitments()
        assert multi[0].get_token_index() == 0
