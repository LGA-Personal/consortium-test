"""
Audit Orchestration

The Auditor is responsible for:
- Selecting work units to audit based on probability
- Dispatching audit orders to verifier nodes
- Collecting and processing audit results
- Detecting fraud and triggering appropriate responses
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from consortium.core.session import Session, WorkUnit
from consortium.verification.canonicalizer import verify_commitment

logger = logging.getLogger(__name__)


class AuditStatus(Enum):
    """Status of an audit."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class AuditRecord:
    """Record of a single audit."""

    audit_id: str
    session_id: str
    work_order_id: str
    token_index: int
    stage_id: int
    original_node_id: str
    verifier_node_id: str
    expected_commitment: bytes
    input_activation: np.ndarray
    status: AuditStatus = AuditStatus.PENDING
    computed_commitment: Optional[bytes] = None
    failure_reason: Optional[str] = None
    scheduled_at_ms: int = 0
    completed_at_ms: int = 0
    verify_time_us: int = 0


@dataclass
class AuditStats:
    """Statistics for audit results."""

    total_audits: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    timeouts: int = 0
    average_verify_time_us: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_audits == 0:
            return 1.0
        return self.passed / self.total_audits

    @property
    def fraud_detected(self) -> bool:
        """Check if any fraud was detected."""
        return self.failed > 0


class Auditor:
    """
    Orchestrates audit selection and execution.

    The auditor randomly selects work units for verification based on
    the configured audit probability. Selected work is dispatched to
    verifier nodes (nodes that can recompute the work and compare
    commitments).
    """

    def __init__(
        self,
        audit_probability: float = 0.2,
        audit_timeout_ms: int = 30000,
        on_fraud_detected: Optional[Callable[[AuditRecord], None]] = None,
    ):
        """
        Initialize the auditor.

        Args:
            audit_probability: Probability of auditing each work unit (0.0-1.0)
            audit_timeout_ms: Timeout for audit completion
            on_fraud_detected: Callback when fraud is detected
        """
        self.audit_probability = audit_probability
        self.audit_timeout_ms = audit_timeout_ms
        self.on_fraud_detected = on_fraud_detected

        self.pending_audits: Dict[str, AuditRecord] = {}
        self.completed_audits: List[AuditRecord] = []
        self.stats = AuditStats()

        self._rng = np.random.default_rng(42)

    def should_audit(self, session: Optional[Session] = None) -> bool:
        """
        Determine if a work unit should be audited.

        Uses the session's RNG if provided for deterministic audit selection.
        """
        if session is not None:
            return session.should_audit()
        return self._rng.random() < self.audit_probability

    def schedule_audit(
        self,
        session_id: str,
        work_order_id: str,
        token_index: int,
        stage_id: int,
        original_node_id: str,
        verifier_node_id: str,
        expected_commitment: bytes,
        input_activation: np.ndarray,
    ) -> str:
        """
        Schedule an audit for a completed work unit.

        Args:
            session_id: Session ID
            work_order_id: Original work order ID
            token_index: Token index being processed
            stage_id: Pipeline stage
            original_node_id: Node that did the original work
            verifier_node_id: Node that will verify
            expected_commitment: Expected commitment hash
            input_activation: Input to recompute

        Returns:
            Audit ID
        """
        audit_id = str(uuid.uuid4())

        record = AuditRecord(
            audit_id=audit_id,
            session_id=session_id,
            work_order_id=work_order_id,
            token_index=token_index,
            stage_id=stage_id,
            original_node_id=original_node_id,
            verifier_node_id=verifier_node_id,
            expected_commitment=expected_commitment,
            input_activation=input_activation,
            status=AuditStatus.PENDING,
            scheduled_at_ms=int(time.time() * 1000),
        )

        self.pending_audits[audit_id] = record
        self.stats.total_audits += 1

        logger.debug(
            f"Scheduled audit {audit_id[:8]} for token {token_index}, "
            f"stage {stage_id} (verifier: {verifier_node_id})"
        )

        return audit_id

    def mark_in_progress(self, audit_id: str) -> None:
        """Mark an audit as in progress."""
        if audit_id in self.pending_audits:
            self.pending_audits[audit_id].status = AuditStatus.IN_PROGRESS

    def complete_audit(
        self,
        audit_id: str,
        passed: bool,
        computed_commitment: Optional[bytes] = None,
        failure_reason: Optional[str] = None,
        verify_time_us: int = 0,
    ) -> None:
        """
        Complete an audit with results.

        Args:
            audit_id: Audit ID
            passed: Whether the audit passed
            computed_commitment: Commitment computed by verifier
            failure_reason: Reason for failure (if failed)
            verify_time_us: Time to verify
        """
        if audit_id not in self.pending_audits:
            logger.warning(f"Unknown audit ID: {audit_id}")
            return

        record = self.pending_audits.pop(audit_id)
        record.completed_at_ms = int(time.time() * 1000)
        record.computed_commitment = computed_commitment
        record.verify_time_us = verify_time_us

        if passed:
            record.status = AuditStatus.PASSED
            self.stats.passed += 1
            logger.info(f"Audit {audit_id[:8]} PASSED")
        else:
            record.status = AuditStatus.FAILED
            record.failure_reason = failure_reason
            self.stats.failed += 1
            logger.warning(
                f"Audit {audit_id[:8]} FAILED: {failure_reason}"
            )

            # Trigger fraud callback
            if self.on_fraud_detected:
                self.on_fraud_detected(record)

        self.completed_audits.append(record)

        # Update average verify time
        total_time = sum(a.verify_time_us for a in self.completed_audits)
        self.stats.average_verify_time_us = total_time / len(self.completed_audits)

    def mark_timeout(self, audit_id: str) -> None:
        """Mark an audit as timed out."""
        if audit_id not in self.pending_audits:
            return

        record = self.pending_audits.pop(audit_id)
        record.status = AuditStatus.TIMEOUT
        record.failure_reason = "Audit timed out"
        record.completed_at_ms = int(time.time() * 1000)

        self.stats.timeouts += 1
        self.completed_audits.append(record)

        logger.warning(f"Audit {audit_id[:8]} timed out")

    def mark_error(self, audit_id: str, error: str) -> None:
        """Mark an audit as errored."""
        if audit_id not in self.pending_audits:
            return

        record = self.pending_audits.pop(audit_id)
        record.status = AuditStatus.ERROR
        record.failure_reason = error
        record.completed_at_ms = int(time.time() * 1000)

        self.stats.errors += 1
        self.completed_audits.append(record)

        logger.error(f"Audit {audit_id[:8]} error: {error}")

    def check_timeouts(self) -> List[str]:
        """Check for timed out audits and mark them."""
        now_ms = int(time.time() * 1000)
        timed_out = []

        for audit_id, record in list(self.pending_audits.items()):
            elapsed = now_ms - record.scheduled_at_ms
            if elapsed > self.audit_timeout_ms:
                self.mark_timeout(audit_id)
                timed_out.append(audit_id)

        return timed_out

    def get_pending_count(self) -> int:
        """Get number of pending audits."""
        return len(self.pending_audits)

    def get_stats(self) -> dict:
        """Get audit statistics."""
        return {
            "total_audits": self.stats.total_audits,
            "passed": self.stats.passed,
            "failed": self.stats.failed,
            "errors": self.stats.errors,
            "timeouts": self.stats.timeouts,
            "pending": len(self.pending_audits),
            "pass_rate": self.stats.pass_rate,
            "fraud_detected": self.stats.fraud_detected,
            "average_verify_time_us": self.stats.average_verify_time_us,
        }

    def verify_locally(
        self,
        input_activation: np.ndarray,
        expected_commitment: bytes,
        forward_fn: Callable[[np.ndarray], np.ndarray],
    ) -> bool:
        """
        Verify a work unit locally (for testing or single-node mode).

        Args:
            input_activation: Input to recompute
            expected_commitment: Expected commitment
            forward_fn: Function to execute forward pass

        Returns:
            True if verification passes
        """
        # Recompute
        output = forward_fn(input_activation)

        # Verify commitment
        return verify_commitment(output, expected_commitment)


def select_verifier_node(
    original_node_id: str,
    available_nodes: List[str],
    node_capabilities: Dict[str, List[str]],
) -> Optional[str]:
    """
    Select a verifier node for an audit.

    The verifier must:
    1. Be different from the original worker
    2. Have the "verify" capability
    3. Be healthy

    Args:
        original_node_id: Node that did the original work
        available_nodes: List of available node IDs
        node_capabilities: Map of node_id -> capabilities

    Returns:
        Selected verifier node ID, or None if no suitable verifier
    """
    for node_id in available_nodes:
        if node_id == original_node_id:
            continue

        capabilities = node_capabilities.get(node_id, [])
        if "verify" in capabilities:
            return node_id

    return None
