"""
Session Management

This module handles inference session state including:
- Session configuration and lifecycle
- Stage placements
- Token tracking
- Receipt collection
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class SessionStatus(Enum):
    """Session lifecycle states."""

    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class StagePlacement:
    """Assignment of a pipeline stage to a node."""

    stage_id: int
    node_id: str
    layer_start: int  # Inclusive
    layer_end: int  # Exclusive

    @property
    def num_layers(self) -> int:
        """Number of layers in this stage."""
        return self.layer_end - self.layer_start

    def __post_init__(self):
        if self.layer_end <= self.layer_start:
            raise ValueError(
                f"layer_end ({self.layer_end}) must be > layer_start ({self.layer_start})"
            )


@dataclass
class SessionConfig:
    """Configuration for an inference session."""

    model_id: str = "llama3-8b-q4_k_m"
    rng_seed: int = 42
    audit_probability: float = 0.2
    max_tokens: int = 64
    temperature: float = 0.0  # 0.0 for greedy
    top_k: int = 1  # 1 for greedy

    # Timeouts
    stage_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 5000

    # Model architecture (Llama-3-8B defaults)
    num_layers: int = 32
    hidden_dim: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    vocab_size: int = 128256

    # Verification settings (EXO integration)
    verification_enabled: bool = True
    verification_interval: int = 1  # Verify every N-th layer (1 = all layers)
    verification_grid_factor: int = 64
    verification_clamp_min: float = -100.0
    verification_clamp_max: float = 100.0

    def __post_init__(self):
        if not 0.0 <= self.audit_probability <= 1.0:
            raise ValueError(
                f"audit_probability must be in [0, 1], got {self.audit_probability}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.verification_interval < 1:
            raise ValueError(
                f"verification_interval must be >= 1, got {self.verification_interval}"
            )


@dataclass
class LayerCommitmentBatch:
    """
    Batch of per-layer commitments from a single device for a single token.

    Used for EXO integration where each device reports commitments for
    all layers it processed during token generation.
    """
    token_index: int
    device_rank: int
    commitments: Dict[int, bytes]  # layer_idx -> 32-byte SHA-256
    timestamp_ms: int
    command_id: str = ""
    audited: bool = False
    audit_passed: Optional[bool] = None


@dataclass
class PendingAudit:
    """
    An audit that has been scheduled but not yet completed.
    """
    audit_id: str
    session_id: str
    token_index: int
    layer_idx: int
    device_rank: int
    expected_commitment: bytes
    verifier_node_id: str
    scheduled_at_ms: int
    completed_at_ms: Optional[int] = None
    passed: Optional[bool] = None
    computed_commitment: Optional[bytes] = None
    failure_reason: Optional[str] = None


@dataclass
class Receipt:
    """Signed receipt for completed work."""

    session_id: str
    order_id: str
    node_id: str
    token_index: int
    stage_id: int
    commitment: bytes
    input_hash: bytes
    timestamp_ms: int
    signature: bytes


@dataclass
class WorkUnit:
    """Tracking for a single unit of work (one stage of one token)."""

    order_id: str
    token_index: int
    stage_id: int
    input_activation: Optional[np.ndarray] = None
    output_activation: Optional[np.ndarray] = None
    commitment: Optional[bytes] = None
    receipt: Optional[Receipt] = None
    compute_time_us: int = 0
    audited: bool = False
    audit_passed: Optional[bool] = None


@dataclass
class TokenGeneration:
    """State for generating a single token."""

    token_index: int
    work_units: Dict[int, WorkUnit] = field(default_factory=dict)  # stage_id -> WorkUnit
    final_logits: Optional[np.ndarray] = None
    sampled_token: Optional[int] = None
    total_time_ms: float = 0.0

    @property
    def is_complete(self) -> bool:
        """Check if all stages have completed."""
        return self.sampled_token is not None


@dataclass
class Session:
    """
    Inference session state.

    Tracks the complete state of a distributed inference session including
    configuration, placements, generated tokens, and receipts.
    """

    id: str
    config: SessionConfig
    placements: List[StagePlacement]
    prompt_tokens: List[int] = field(default_factory=list)
    generated_tokens: List[int] = field(default_factory=list)
    token_generations: List[TokenGeneration] = field(default_factory=list)
    receipts: List[Receipt] = field(default_factory=list)
    status: SessionStatus = SessionStatus.CREATED
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    started_at_ms: Optional[int] = None
    completed_at_ms: Optional[int] = None
    error_message: Optional[str] = None

    # Layer commitments from EXO integration (M5)
    # Key: (token_index, device_rank) -> LayerCommitmentBatch
    layer_commitments: Dict[tuple, "LayerCommitmentBatch"] = field(default_factory=dict)

    # Pending audits for layer commitments
    pending_layer_audits: Dict[str, PendingAudit] = field(default_factory=dict)

    # Completed layer audits
    completed_layer_audits: List[PendingAudit] = field(default_factory=list)

    # Random state for deterministic sampling and audit selection
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )

    @classmethod
    def create(
        cls,
        config: SessionConfig,
        placements: List[StagePlacement],
        prompt_tokens: List[int],
    ) -> "Session":
        """Create a new session with the given configuration."""
        session = cls(
            id=str(uuid.uuid4()),
            config=config,
            placements=placements,
            prompt_tokens=prompt_tokens,
            rng=np.random.default_rng(config.rng_seed),
        )
        return session

    @property
    def num_stages(self) -> int:
        """Number of pipeline stages."""
        return len(self.placements)

    @property
    def total_tokens_generated(self) -> int:
        """Number of tokens generated so far."""
        return len(self.generated_tokens)

    @property
    def is_complete(self) -> bool:
        """Check if generation is complete."""
        return (
            self.status == SessionStatus.COMPLETED
            or len(self.generated_tokens) >= self.config.max_tokens
        )

    @property
    def total_work_units(self) -> int:
        """Total number of work units (tokens * stages)."""
        return self.total_tokens_generated * self.num_stages

    def get_placement_for_stage(self, stage_id: int) -> StagePlacement:
        """Get placement for a specific stage."""
        for placement in self.placements:
            if placement.stage_id == stage_id:
                return placement
        raise ValueError(f"No placement found for stage {stage_id}")

    def get_placement_for_node(self, node_id: str) -> Optional[StagePlacement]:
        """Get placement assigned to a specific node."""
        for placement in self.placements:
            if placement.node_id == node_id:
                return placement
        return None

    def should_audit(self) -> bool:
        """Determine if the current work unit should be audited."""
        return self.rng.random() < self.config.audit_probability

    def add_receipt(self, receipt: Receipt) -> None:
        """Add a receipt to the session."""
        self.receipts.append(receipt)

    def start(self) -> None:
        """Mark session as started."""
        self.status = SessionStatus.RUNNING
        self.started_at_ms = int(time.time() * 1000)

    def complete(self) -> None:
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.completed_at_ms = int(time.time() * 1000)

    def fail(self, error: str) -> None:
        """Mark session as failed."""
        self.status = SessionStatus.FAILED
        self.error_message = error
        self.completed_at_ms = int(time.time() * 1000)

    def abort(self, reason: str) -> None:
        """Abort the session."""
        self.status = SessionStatus.ABORTED
        self.error_message = reason
        self.completed_at_ms = int(time.time() * 1000)

    def get_duration_ms(self) -> Optional[float]:
        """Get session duration in milliseconds."""
        if self.started_at_ms is None:
            return None
        end_time = self.completed_at_ms or int(time.time() * 1000)
        return end_time - self.started_at_ms

    def get_tokens_per_second(self) -> Optional[float]:
        """Calculate token generation rate."""
        duration_ms = self.get_duration_ms()
        if duration_ms is None or duration_ms == 0:
            return None
        return (self.total_tokens_generated / duration_ms) * 1000

    def get_audit_summary(self) -> Dict[str, int]:
        """Get summary of audit results."""
        audited = [tg for tg in self.token_generations for wu in tg.work_units.values() if wu.audited]
        passed = sum(1 for tg in self.token_generations for wu in tg.work_units.values() if wu.audit_passed is True)
        failed = sum(1 for tg in self.token_generations for wu in tg.work_units.values() if wu.audit_passed is False)
        return {
            "total_audited": len(audited),
            "passed": passed,
            "failed": failed,
        }

    # ========== Layer Commitment Methods (EXO Integration) ==========

    def add_layer_commitments(
        self,
        token_index: int,
        device_rank: int,
        commitments: Dict[int, bytes],
        timestamp_ms: int,
        command_id: str = "",
    ) -> LayerCommitmentBatch:
        """
        Add layer commitments from an EXO device.

        Args:
            token_index: Index of the generated token
            device_rank: Rank of the device that computed these layers
            commitments: Mapping from layer_idx to 32-byte commitment
            timestamp_ms: Timestamp when commitments were generated
            command_id: EXO command ID for correlation

        Returns:
            The created LayerCommitmentBatch
        """
        batch = LayerCommitmentBatch(
            token_index=token_index,
            device_rank=device_rank,
            commitments=commitments,
            timestamp_ms=timestamp_ms,
            command_id=command_id,
        )
        key = (token_index, device_rank)
        self.layer_commitments[key] = batch
        return batch

    def get_layer_commitments(
        self,
        token_index: int,
        device_rank: Optional[int] = None,
    ) -> List[LayerCommitmentBatch]:
        """
        Get layer commitments for a token.

        Args:
            token_index: Token index to query
            device_rank: Optional device rank filter

        Returns:
            List of LayerCommitmentBatch matching the criteria
        """
        results = []
        for (t_idx, d_rank), batch in self.layer_commitments.items():
            if t_idx == token_index:
                if device_rank is None or d_rank == device_rank:
                    results.append(batch)
        return results

    def select_layers_for_audit(
        self,
        token_index: int,
    ) -> List[tuple]:
        """
        Select layers to audit based on audit_probability.

        Randomly samples layers from all devices for a given token
        according to the session's audit probability.

        Args:
            token_index: Token index to select audits for

        Returns:
            List of (device_rank, layer_idx) tuples to audit
        """
        layers_to_audit = []

        for (t_idx, device_rank), batch in self.layer_commitments.items():
            if t_idx != token_index:
                continue

            for layer_idx in batch.commitments.keys():
                if self.rng.random() < self.config.audit_probability:
                    layers_to_audit.append((device_rank, layer_idx))

        return layers_to_audit

    def schedule_layer_audit(
        self,
        audit_id: str,
        token_index: int,
        layer_idx: int,
        device_rank: int,
        verifier_node_id: str,
    ) -> Optional[PendingAudit]:
        """
        Schedule an audit for a specific layer commitment.

        Args:
            audit_id: Unique ID for this audit
            token_index: Token that was generated
            layer_idx: Layer to re-verify
            device_rank: Device that originally computed this layer
            verifier_node_id: Node that will perform verification

        Returns:
            PendingAudit if commitment exists, None otherwise
        """
        key = (token_index, device_rank)
        batch = self.layer_commitments.get(key)

        if batch is None or layer_idx not in batch.commitments:
            return None

        audit = PendingAudit(
            audit_id=audit_id,
            session_id=self.id,
            token_index=token_index,
            layer_idx=layer_idx,
            device_rank=device_rank,
            expected_commitment=batch.commitments[layer_idx],
            verifier_node_id=verifier_node_id,
            scheduled_at_ms=int(time.time() * 1000),
        )

        self.pending_layer_audits[audit_id] = audit
        return audit

    def complete_layer_audit(
        self,
        audit_id: str,
        passed: bool,
        computed_commitment: Optional[bytes] = None,
        failure_reason: Optional[str] = None,
    ) -> Optional[PendingAudit]:
        """
        Complete a pending layer audit.

        Args:
            audit_id: ID of the audit to complete
            passed: Whether the audit passed
            computed_commitment: The recomputed commitment
            failure_reason: Reason for failure if passed=False

        Returns:
            The completed audit, or None if not found
        """
        audit = self.pending_layer_audits.pop(audit_id, None)
        if audit is None:
            return None

        audit.completed_at_ms = int(time.time() * 1000)
        audit.passed = passed
        audit.computed_commitment = computed_commitment
        audit.failure_reason = failure_reason

        # Mark the batch as audited
        key = (audit.token_index, audit.device_rank)
        if key in self.layer_commitments:
            batch = self.layer_commitments[key]
            batch.audited = True
            if not passed:
                batch.audit_passed = False
            elif batch.audit_passed is None:
                batch.audit_passed = True

        self.completed_layer_audits.append(audit)
        return audit

    def get_layer_audit_summary(self) -> Dict[str, int]:
        """Get summary of layer-level audit results."""
        total_layers = sum(
            len(batch.commitments) for batch in self.layer_commitments.values()
        )
        pending = len(self.pending_layer_audits)
        completed = len(self.completed_layer_audits)
        passed = sum(1 for a in self.completed_layer_audits if a.passed)
        failed = sum(1 for a in self.completed_layer_audits if not a.passed)

        return {
            "total_layers": total_layers,
            "total_batches": len(self.layer_commitments),
            "pending_audits": pending,
            "completed_audits": completed,
            "passed": passed,
            "failed": failed,
        }

    def to_dict(self) -> dict:
        """Convert session state to dictionary for serialization."""
        return {
            "id": self.id,
            "status": self.status.value,
            "config": {
                "model_id": self.config.model_id,
                "max_tokens": self.config.max_tokens,
                "audit_probability": self.config.audit_probability,
                "rng_seed": self.config.rng_seed,
            },
            "placements": [
                {
                    "stage_id": p.stage_id,
                    "node_id": p.node_id,
                    "layer_start": p.layer_start,
                    "layer_end": p.layer_end,
                }
                for p in self.placements
            ],
            "tokens_generated": self.total_tokens_generated,
            "receipts_collected": len(self.receipts),
            "duration_ms": self.get_duration_ms(),
            "tokens_per_second": self.get_tokens_per_second(),
            "audit_summary": self.get_audit_summary(),
            "layer_audit_summary": self.get_layer_audit_summary(),
        }


def create_default_placements(
    num_layers: int = 32,
    num_stages: int = 3,
) -> List[StagePlacement]:
    """
    Create default stage placements for a model.

    Splits layers roughly equally across stages.
    Node IDs are placeholders until actual nodes register.
    """
    layers_per_stage = num_layers // num_stages
    extra_layers = num_layers % num_stages

    placements = []
    current_layer = 0

    for stage_id in range(num_stages):
        # Distribute extra layers to earlier stages
        stage_layers = layers_per_stage + (1 if stage_id < extra_layers else 0)
        layer_end = current_layer + stage_layers

        placements.append(
            StagePlacement(
                stage_id=stage_id,
                node_id=f"node-{stage_id}",  # Placeholder
                layer_start=current_layer,
                layer_end=layer_end,
            )
        )
        current_layer = layer_end

    return placements
