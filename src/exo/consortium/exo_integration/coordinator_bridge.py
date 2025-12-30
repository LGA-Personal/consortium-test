"""
Coordinator Bridge for EXO Integration

Bridges between EXO's event system and the consortium coordinator.
Receives LayerCommitmentsGenerated events from EXO runners and
forwards them to the consortium coordinator via gRPC.

Usage:
    ```python
    bridge = CoordinatorBridge(
        coordinator_address="localhost:50051",
        session_id="session-123",
    )

    # Forward an event from EXO
    event = LayerCommitmentsGenerated(...)
    await bridge.forward_commitments(event)

    # Or use as an event handler
    event_receiver.subscribe(bridge.handle_event)
    ```
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import TYPE_CHECKING, Dict, Optional

import grpc

from exo.consortium.transport import consortium_pb2, consortium_pb2_grpc

if TYPE_CHECKING:
    from exo.shared.types.events import LayerCommitmentsGenerated

logger = logging.getLogger(__name__)


@dataclass
class CommitmentSubmission:
    """A commitment submission queued for delivery."""

    session_id: str
    command_id: str
    token_index: int
    device_rank: int
    commitments: Dict[int, bytes]
    timestamp_ms: int
    retry_count: int = 0
    max_retries: int = 3


class CoordinatorBridge:
    """
    Bridges EXO commitment events to the consortium coordinator.

    This class handles the asynchronous submission of layer commitments
    to the coordinator, with retry logic and graceful error handling.
    """

    def __init__(
        self,
        coordinator_address: str = "localhost:50051",
        session_id: Optional[str] = None,
        async_mode: bool = True,
        retry_interval_ms: int = 1000,
    ):
        """
        Initialize the bridge.

        Args:
            coordinator_address: gRPC address of the coordinator
            session_id: Default session ID (can be overridden per event)
            async_mode: If True, queue submissions for background processing
            retry_interval_ms: Time between retries on failure
        """
        self._coordinator_address = coordinator_address
        self._session_id = session_id
        self._async_mode = async_mode
        self._retry_interval_ms = retry_interval_ms

        # gRPC channel and stub
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[consortium_pb2_grpc.CoordinatorServiceStub] = None

        # Async queue for background processing
        self._queue: Queue[CommitmentSubmission] = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # Statistics
        self._submissions_sent = 0
        self._submissions_failed = 0

    def start(self) -> None:
        """Start the bridge (connects to coordinator and starts worker)."""
        if self._running:
            return

        # Create gRPC channel
        self._channel = grpc.insecure_channel(self._coordinator_address)
        self._stub = consortium_pb2_grpc.CoordinatorServiceStub(self._channel)

        if self._async_mode:
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="CoordinatorBridge-Worker",
            )
            self._worker_thread.start()
            logger.info(
                f"CoordinatorBridge started (async mode, connecting to {self._coordinator_address})"
            )
        else:
            logger.info(
                f"CoordinatorBridge started (sync mode, connecting to {self._coordinator_address})"
            )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the bridge gracefully."""
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None

        logger.info(
            f"CoordinatorBridge stopped. Sent: {self._submissions_sent}, Failed: {self._submissions_failed}"
        )

    def forward_commitments(
        self,
        event: "LayerCommitmentsGenerated",
        session_id: Optional[str] = None,
    ) -> None:
        """
        Forward a LayerCommitmentsGenerated event to the coordinator.

        Args:
            event: The commitment event from EXO
            session_id: Override session ID (uses default if not provided)
        """
        effective_session_id = session_id or self._session_id or ""

        submission = CommitmentSubmission(
            session_id=effective_session_id,
            command_id=event.command_id,
            token_index=event.token_index,
            device_rank=event.device_rank,
            commitments=event.commitments,
            timestamp_ms=event.timestamp_ms,
        )

        if self._async_mode:
            self._queue.put(submission)
        else:
            self._send_submission(submission)

    def handle_event(self, event: object) -> None:
        """
        Handle an event from EXO's event stream.

        This method can be used as an event handler callback.
        Only processes LayerCommitmentsGenerated events.

        Args:
            event: Any EXO event
        """
        # Import here to avoid circular imports
        try:
            from exo.shared.types.events import LayerCommitmentsGenerated

            if isinstance(event, LayerCommitmentsGenerated):
                self.forward_commitments(event)
        except ImportError:
            logger.warning("EXO types not available for event handling")

    def _worker_loop(self) -> None:
        """Background worker that processes the submission queue."""
        while self._running:
            try:
                submission = self._queue.get(timeout=0.5)
                self._send_submission(submission)
            except Empty:
                continue
            except Exception as e:
                logger.exception(f"Worker error: {e}")

    def _send_submission(self, submission: CommitmentSubmission) -> bool:
        """
        Send a commitment submission to the coordinator.

        Returns:
            True if successful, False otherwise
        """
        if self._stub is None:
            logger.error("Bridge not started - cannot send submission")
            return False

        # Build protobuf message
        layer_commitments = [
            consortium_pb2.LayerCommitment(layer_idx=idx, commitment=commitment)
            for idx, commitment in submission.commitments.items()
        ]

        request = consortium_pb2.SubmitLayerCommitmentsRequest(
            session_id=submission.session_id,
            command_id=submission.command_id,
            token_index=submission.token_index,
            device_rank=submission.device_rank,
            commitments=layer_commitments,
            timestamp_ms=submission.timestamp_ms,
        )

        try:
            response = self._stub.SubmitLayerCommitments(request, timeout=5.0)

            if response.acknowledged:
                self._submissions_sent += 1
                logger.debug(
                    f"Submitted commitments for token {submission.token_index} "
                    f"({len(submission.commitments)} layers)"
                )
                return True
            else:
                logger.warning(f"Coordinator rejected submission: {response.error}")
                self._submissions_failed += 1
                return False

        except grpc.RpcError as e:
            logger.error(f"gRPC error submitting commitments: {e}")

            # Retry if applicable
            if (
                submission.retry_count < submission.max_retries
                and self._async_mode
                and self._running
            ):
                submission.retry_count += 1
                time.sleep(self._retry_interval_ms / 1000)
                self._queue.put(submission)
            else:
                self._submissions_failed += 1

            return False

    def set_session_id(self, session_id: str) -> None:
        """Set the default session ID for all submissions."""
        self._session_id = session_id

    @property
    def is_running(self) -> bool:
        """Check if the bridge is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def stats(self) -> Dict[str, int]:
        """Get submission statistics."""
        return {
            "sent": self._submissions_sent,
            "failed": self._submissions_failed,
            "queued": self._queue.qsize(),
        }


class AsyncCoordinatorBridge:
    """
    Async version of CoordinatorBridge for asyncio-based applications.

    Uses grpc.aio for async gRPC calls.
    """

    def __init__(
        self,
        coordinator_address: str = "localhost:50051",
        session_id: Optional[str] = None,
    ):
        self._coordinator_address = coordinator_address
        self._session_id = session_id
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[consortium_pb2_grpc.CoordinatorServiceStub] = None
        self._submissions_sent = 0
        self._submissions_failed = 0

    async def start(self) -> None:
        """Start the async bridge."""
        self._channel = grpc.aio.insecure_channel(self._coordinator_address)
        self._stub = consortium_pb2_grpc.CoordinatorServiceStub(self._channel)
        logger.info(f"AsyncCoordinatorBridge started ({self._coordinator_address})")

    async def stop(self) -> None:
        """Stop the async bridge."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
        logger.info(
            f"AsyncCoordinatorBridge stopped. Sent: {self._submissions_sent}, Failed: {self._submissions_failed}"
        )

    async def forward_commitments(
        self,
        event: "LayerCommitmentsGenerated",
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Asynchronously forward commitments to the coordinator.

        Returns:
            True if successful, False otherwise
        """
        if self._stub is None:
            logger.error("Bridge not started")
            return False

        effective_session_id = session_id or self._session_id or ""

        layer_commitments = [
            consortium_pb2.LayerCommitment(layer_idx=idx, commitment=commitment)
            for idx, commitment in event.commitments.items()
        ]

        request = consortium_pb2.SubmitLayerCommitmentsRequest(
            session_id=effective_session_id,
            command_id=event.command_id,
            token_index=event.token_index,
            device_rank=event.device_rank,
            commitments=layer_commitments,
            timestamp_ms=event.timestamp_ms,
        )

        try:
            response = await self._stub.SubmitLayerCommitments(request, timeout=5.0)

            if response.acknowledged:
                self._submissions_sent += 1
                return True
            else:
                logger.warning(f"Coordinator rejected: {response.error}")
                self._submissions_failed += 1
                return False

        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e}")
            self._submissions_failed += 1
            return False

    def set_session_id(self, session_id: str) -> None:
        """Set the default session ID."""
        self._session_id = session_id

    @property
    def stats(self) -> Dict[str, int]:
        """Get submission statistics."""
        return {
            "sent": self._submissions_sent,
            "failed": self._submissions_failed,
        }
