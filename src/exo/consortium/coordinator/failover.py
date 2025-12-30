"""
Failover and Recovery Logic

The FailoverManager is responsible for:
- Detecting node failures (via heartbeat timeouts)
- Selecting backup nodes for failed stages
- Coordinating KV cache recovery
- Resuming inference from the last checkpoint
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from exo.consortium.core.session import Session, StagePlacement, SessionStatus

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Type of node failure."""

    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    WORK_TIMEOUT = "work_timeout"
    CONNECTION_ERROR = "connection_error"
    EXPLICIT_FAILURE = "explicit_failure"


class FailoverStatus(Enum):
    """Status of a failover operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FailureEvent:
    """Record of a node failure."""

    event_id: str
    session_id: str
    failed_node_id: str
    failed_stage_id: int
    failure_type: FailureType
    detected_at_ms: int
    last_successful_token: int
    last_successful_order_id: Optional[str] = None


@dataclass
class FailoverOperation:
    """Record of a failover operation."""

    failover_id: str
    failure_event: FailureEvent
    backup_node_id: str
    status: FailoverStatus = FailoverStatus.PENDING
    started_at_ms: int = 0
    completed_at_ms: int = 0
    resume_token_index: int = 0
    error_message: Optional[str] = None
    kv_cache_restored: bool = False
    checkpoint_position: int = 0


@dataclass
class FailoverStats:
    """Statistics for failover operations."""

    total_failures: int = 0
    successful_failovers: int = 0
    failed_failovers: int = 0
    average_recovery_time_ms: float = 0.0
    total_tokens_recomputed: int = 0


class FailoverManager:
    """
    Manages node failure detection and recovery.

    The failover process:
    1. Detect failure (timeout or explicit)
    2. Select backup node capable of handling the failed stage
    3. Transfer last known input activation to backup
    4. Restore KV cache from checkpoint (if available)
    5. Resume computation from the failure point
    """

    def __init__(
        self,
        heartbeat_timeout_ms: int = 15000,
        work_timeout_ms: int = 30000,
        on_failure_detected: Optional[Callable[[FailureEvent], None]] = None,
        on_failover_complete: Optional[Callable[[FailoverOperation], None]] = None,
    ):
        """
        Initialize the failover manager.

        Args:
            heartbeat_timeout_ms: Timeout for node heartbeat
            work_timeout_ms: Timeout for work completion
            on_failure_detected: Callback when failure is detected
            on_failover_complete: Callback when failover completes
        """
        self.heartbeat_timeout_ms = heartbeat_timeout_ms
        self.work_timeout_ms = work_timeout_ms
        self.on_failure_detected = on_failure_detected
        self.on_failover_complete = on_failover_complete

        self.failure_events: List[FailureEvent] = []
        self.pending_failovers: Dict[str, FailoverOperation] = {}
        self.completed_failovers: List[FailoverOperation] = []
        self.failed_nodes: Set[str] = set()
        self.stats = FailoverStats()

    def detect_failure(
        self,
        session_id: str,
        failed_node_id: str,
        failed_stage_id: int,
        failure_type: FailureType,
        last_successful_token: int,
        last_successful_order_id: Optional[str] = None,
    ) -> FailureEvent:
        """
        Record a detected node failure.

        Args:
            session_id: Session ID
            failed_node_id: ID of the failed node
            failed_stage_id: Stage that was being processed
            failure_type: Type of failure
            last_successful_token: Last token that completed successfully
            last_successful_order_id: Last successful work order ID

        Returns:
            FailureEvent record
        """
        event = FailureEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            failed_node_id=failed_node_id,
            failed_stage_id=failed_stage_id,
            failure_type=failure_type,
            detected_at_ms=int(time.time() * 1000),
            last_successful_token=last_successful_token,
            last_successful_order_id=last_successful_order_id,
        )

        self.failure_events.append(event)
        self.failed_nodes.add(failed_node_id)
        self.stats.total_failures += 1

        logger.warning(
            f"Failure detected: node={failed_node_id}, stage={failed_stage_id}, "
            f"type={failure_type.value}, last_token={last_successful_token}"
        )

        if self.on_failure_detected:
            self.on_failure_detected(event)

        return event

    def select_backup_node(
        self,
        failed_stage_id: int,
        session: Session,
        available_nodes: List[str],
        node_capabilities: Dict[str, List[str]],
    ) -> Optional[str]:
        """
        Select a backup node for a failed stage.

        Selection criteria:
        1. Node must be healthy (not failed)
        2. Node must have compute capability
        3. Prefer nodes not already assigned to other stages
        4. Prefer coordinator node as universal backup

        Args:
            failed_stage_id: Stage that failed
            session: Current session
            available_nodes: List of available node IDs
            node_capabilities: Map of node_id -> capabilities

        Returns:
            Selected backup node ID, or None if no suitable backup
        """
        # Get nodes already assigned to stages
        assigned_nodes = {p.node_id for p in session.placements}

        # First, try to find an unassigned node with compute capability
        for node_id in available_nodes:
            if node_id in self.failed_nodes:
                continue
            if node_id in assigned_nodes and node_id != session.placements[-1].node_id:
                continue
            capabilities = node_capabilities.get(node_id, [])
            if "compute" in capabilities:
                return node_id

        # Fall back to coordinator (last stage node) as universal backup
        coordinator_id = session.placements[-1].node_id
        if coordinator_id not in self.failed_nodes:
            return coordinator_id

        logger.error("No suitable backup node available")
        return None

    def initiate_failover(
        self,
        failure_event: FailureEvent,
        backup_node_id: str,
        resume_token_index: int,
    ) -> FailoverOperation:
        """
        Initiate a failover operation.

        Args:
            failure_event: The failure that triggered failover
            backup_node_id: Selected backup node
            resume_token_index: Token index to resume from

        Returns:
            FailoverOperation record
        """
        failover = FailoverOperation(
            failover_id=str(uuid.uuid4()),
            failure_event=failure_event,
            backup_node_id=backup_node_id,
            status=FailoverStatus.PENDING,
            started_at_ms=int(time.time() * 1000),
            resume_token_index=resume_token_index,
        )

        self.pending_failovers[failover.failover_id] = failover

        logger.info(
            f"Initiated failover {failover.failover_id[:8]}: "
            f"stage {failure_event.failed_stage_id} -> {backup_node_id}, "
            f"resume at token {resume_token_index}"
        )

        return failover

    def mark_failover_in_progress(self, failover_id: str) -> None:
        """Mark a failover as in progress."""
        if failover_id in self.pending_failovers:
            self.pending_failovers[failover_id].status = FailoverStatus.IN_PROGRESS

    def complete_failover(
        self,
        failover_id: str,
        success: bool,
        kv_cache_restored: bool = False,
        checkpoint_position: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Complete a failover operation.

        Args:
            failover_id: Failover operation ID
            success: Whether failover succeeded
            kv_cache_restored: Whether KV cache was restored from checkpoint
            checkpoint_position: Position of restored checkpoint
            error_message: Error message if failed
        """
        if failover_id not in self.pending_failovers:
            logger.warning(f"Unknown failover ID: {failover_id}")
            return

        failover = self.pending_failovers.pop(failover_id)
        failover.completed_at_ms = int(time.time() * 1000)
        failover.kv_cache_restored = kv_cache_restored
        failover.checkpoint_position = checkpoint_position

        if success:
            failover.status = FailoverStatus.COMPLETED
            self.stats.successful_failovers += 1

            # Calculate recovery time
            recovery_time = failover.completed_at_ms - failover.started_at_ms
            total_time = sum(
                f.completed_at_ms - f.started_at_ms
                for f in self.completed_failovers
                if f.status == FailoverStatus.COMPLETED
            ) + recovery_time
            completed_count = self.stats.successful_failovers
            self.stats.average_recovery_time_ms = total_time / completed_count

            logger.info(
                f"Failover {failover_id[:8]} COMPLETED in {recovery_time}ms "
                f"(KV restored: {kv_cache_restored}, checkpoint pos: {checkpoint_position})"
            )
        else:
            failover.status = FailoverStatus.FAILED
            failover.error_message = error_message
            self.stats.failed_failovers += 1
            logger.error(f"Failover {failover_id[:8]} FAILED: {error_message}")

        self.completed_failovers.append(failover)

        if self.on_failover_complete:
            self.on_failover_complete(failover)

    def update_session_placement(
        self,
        session: Session,
        failed_stage_id: int,
        backup_node_id: str,
    ) -> StagePlacement:
        """
        Update session placement after successful failover.

        Args:
            session: Session to update
            failed_stage_id: Stage that was moved
            backup_node_id: New node handling the stage

        Returns:
            Updated StagePlacement
        """
        # Find and update the placement
        for i, placement in enumerate(session.placements):
            if placement.stage_id == failed_stage_id:
                new_placement = StagePlacement(
                    stage_id=failed_stage_id,
                    node_id=backup_node_id,
                    layer_start=placement.layer_start,
                    layer_end=placement.layer_end,
                )
                session.placements[i] = new_placement

                logger.info(
                    f"Updated placement: stage {failed_stage_id} now on {backup_node_id}"
                )

                return new_placement

        raise ValueError(f"Stage {failed_stage_id} not found in session placements")

    def calculate_recompute_tokens(
        self,
        failure_token: int,
        checkpoint_position: int,
    ) -> int:
        """
        Calculate how many tokens need to be recomputed after failover.

        Args:
            failure_token: Token where failure occurred
            checkpoint_position: Position of last checkpoint

        Returns:
            Number of tokens to recompute
        """
        if checkpoint_position >= failure_token:
            return 0

        tokens_to_recompute = failure_token - checkpoint_position
        self.stats.total_tokens_recomputed += tokens_to_recompute

        return tokens_to_recompute

    def is_node_failed(self, node_id: str) -> bool:
        """Check if a node has failed."""
        return node_id in self.failed_nodes

    def mark_node_recovered(self, node_id: str) -> None:
        """Mark a node as recovered."""
        if node_id in self.failed_nodes:
            self.failed_nodes.remove(node_id)
            logger.info(f"Node {node_id} marked as recovered")

    def get_pending_failovers(self) -> List[FailoverOperation]:
        """Get list of pending failover operations."""
        return list(self.pending_failovers.values())

    def get_stats(self) -> dict:
        """Get failover statistics."""
        return {
            "total_failures": self.stats.total_failures,
            "successful_failovers": self.stats.successful_failovers,
            "failed_failovers": self.stats.failed_failovers,
            "pending_failovers": len(self.pending_failovers),
            "failed_nodes": list(self.failed_nodes),
            "average_recovery_time_ms": self.stats.average_recovery_time_ms,
            "total_tokens_recomputed": self.stats.total_tokens_recomputed,
        }


async def execute_failover(
    failover_manager: FailoverManager,
    failure_event: FailureEvent,
    session: Session,
    available_nodes: List[str],
    node_capabilities: Dict[str, List[str]],
    get_last_input: Callable[[int], Optional[np.ndarray]],
    dispatch_failover_order: Callable[[str, int, int, int, np.ndarray], bool],
    get_checkpoint: Callable[[str], Optional[Tuple[int, bytes]]],
) -> bool:
    """
    Execute a complete failover operation.

    This is the main failover workflow that:
    1. Selects a backup node
    2. Retrieves last input activation
    3. Restores KV cache from checkpoint
    4. Dispatches failover order to backup
    5. Updates session placement

    Args:
        failover_manager: FailoverManager instance
        failure_event: The failure that triggered failover
        session: Current session
        available_nodes: List of available node IDs
        node_capabilities: Map of node_id -> capabilities
        get_last_input: Function to get last input activation for a stage
        dispatch_failover_order: Function to dispatch failover order
        get_checkpoint: Function to get latest checkpoint

    Returns:
        True if failover succeeded
    """
    # Select backup node
    backup_node = failover_manager.select_backup_node(
        failed_stage_id=failure_event.failed_stage_id,
        session=session,
        available_nodes=available_nodes,
        node_capabilities=node_capabilities,
    )

    if backup_node is None:
        logger.error("No backup node available for failover")
        return False

    # Determine resume point
    checkpoint = get_checkpoint(session.id)
    checkpoint_position = checkpoint[0] if checkpoint else 0

    resume_token = max(checkpoint_position, failure_event.last_successful_token)

    # Initiate failover
    failover = failover_manager.initiate_failover(
        failure_event=failure_event,
        backup_node_id=backup_node,
        resume_token_index=resume_token,
    )

    failover_manager.mark_failover_in_progress(failover.failover_id)

    try:
        # Get last input activation
        last_input = get_last_input(failure_event.failed_stage_id)
        if last_input is None:
            raise RuntimeError("Could not retrieve last input activation")

        # Get failed stage placement for layer info
        failed_placement = session.get_placement_for_stage(failure_event.failed_stage_id)

        # Dispatch failover order to backup node
        success = dispatch_failover_order(
            backup_node,
            failed_placement.layer_start,
            failed_placement.layer_end,
            resume_token,
            last_input,
        )

        if not success:
            raise RuntimeError("Failover order dispatch failed")

        # Update session placement
        failover_manager.update_session_placement(
            session=session,
            failed_stage_id=failure_event.failed_stage_id,
            backup_node_id=backup_node,
        )

        # Complete failover
        failover_manager.complete_failover(
            failover_id=failover.failover_id,
            success=True,
            kv_cache_restored=checkpoint is not None,
            checkpoint_position=checkpoint_position,
        )

        return True

    except Exception as e:
        failover_manager.complete_failover(
            failover_id=failover.failover_id,
            success=False,
            error_message=str(e),
        )
        return False
