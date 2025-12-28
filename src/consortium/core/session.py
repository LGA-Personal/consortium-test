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

    def __post_init__(self):
        if not 0.0 <= self.audit_probability <= 1.0:
            raise ValueError(
                f"audit_probability must be in [0, 1], got {self.audit_probability}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")


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
