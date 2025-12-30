"""
In-Process Pipeline Execution

This module provides the core pipeline execution logic without networking.
It orchestrates inference through multiple stages, handling:
- Sequential stage execution
- Activation passing between stages
- Commitment generation
- Receipt creation
- Audit scheduling

This is the foundation that the distributed (gRPC) version builds upon.
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from exo.consortium.core.session import (
    Receipt,
    Session,
    SessionConfig,
    StagePlacement,
    TokenGeneration,
    WorkUnit,
)
from exo.consortium.identity.keys import KeyManager
from exo.consortium.identity.signing import create_and_sign_receipt
from exo.consortium.verification.canonicalizer import compute_commitment
from exo.consortium.worker.kv_cache import StageKVCache
from exo.consortium.worker.model_shard import ForwardResult, ModelShard, ShardConfig

logger = logging.getLogger(__name__)


@dataclass
class StageExecutor:
    """
    Executor for a single pipeline stage.

    Wraps a ModelShard with KV cache management and commitment generation.
    """

    placement: StagePlacement
    shard: ModelShard
    kv_cache: StageKVCache
    key_manager: KeyManager

    @classmethod
    def create(
        cls,
        placement: StagePlacement,
        model_path: Optional[str] = None,
        mock_mode: bool = True,
        hidden_dim: int = 4096,
        num_kv_heads: int = 8,
    ) -> "StageExecutor":
        """Create a stage executor with model shard and KV cache."""
        # Create shard config
        shard_config = ShardConfig(
            model_path=model_path or "",
            layer_start=placement.layer_start,
            layer_end=placement.layer_end,
            hidden_dim=hidden_dim,
            mock_mode=mock_mode,
        )

        # Load model shard
        shard = ModelShard(shard_config)
        shard.load()

        # Create KV cache for this stage
        kv_cache = StageKVCache(
            layer_start=placement.layer_start,
            layer_end=placement.layer_end,
            num_kv_heads=num_kv_heads,
            head_dim=hidden_dim // 32,  # Assuming 32 attention heads
        )

        # Create identity for signing
        key_manager = KeyManager()

        return cls(
            placement=placement,
            shard=shard,
            kv_cache=kv_cache,
            key_manager=key_manager,
        )

    def execute(
        self,
        input_activation: np.ndarray,
        position_offset: int = 0,
    ) -> Tuple[np.ndarray, bytes, int]:
        """
        Execute forward pass through this stage.

        Args:
            input_activation: Input hidden states
            position_offset: Position offset for rotary embeddings

        Returns:
            Tuple of (output_activation, commitment, compute_time_us)
        """
        start_time = time.perf_counter()

        # Execute forward pass
        result = self.shard.forward(input_activation, position_offset)

        # Generate commitment from output
        commitment = compute_commitment(result.hidden_states)

        compute_time_us = int((time.perf_counter() - start_time) * 1_000_000)

        return result.hidden_states, commitment, compute_time_us

    def init_kv_cache(self, batch_size: int = 1) -> None:
        """Initialize KV cache for a new session."""
        self.kv_cache.init_cache(batch_size=batch_size)

    def clear_kv_cache(self) -> None:
        """Clear KV cache."""
        self.kv_cache.clear()


@dataclass
class PipelineResult:
    """Result of a complete pipeline execution for one token."""

    token_index: int
    output_logits: np.ndarray
    commitments: Dict[int, bytes]  # stage_id -> commitment
    compute_times_us: Dict[int, int]  # stage_id -> time
    total_time_ms: float


@dataclass
class InProcessPipeline:
    """
    In-process pipeline executor.

    Executes all stages sequentially without network communication.
    Used for single-node testing and as the foundation for distributed execution.
    """

    session: Session
    executors: Dict[int, StageExecutor] = field(default_factory=dict)
    on_token_callback: Optional[Callable[[int, int], None]] = None  # (token_idx, token_id)
    on_audit_callback: Optional[Callable[[int, int, bool], None]] = None  # (token_idx, stage_id, passed)

    @classmethod
    def create(
        cls,
        config: SessionConfig,
        placements: List[StagePlacement],
        prompt_tokens: List[int],
        model_path: Optional[str] = None,
        mock_mode: bool = True,
    ) -> "InProcessPipeline":
        """
        Create a pipeline with all stages initialized.

        Args:
            config: Session configuration
            placements: Stage-to-layer assignments
            prompt_tokens: Tokenized prompt
            model_path: Path to model file (None for mock mode)
            mock_mode: Use mock forward pass for testing
        """
        # Create session
        session = Session.create(
            config=config,
            placements=placements,
            prompt_tokens=prompt_tokens,
        )

        # Create executors for each stage
        executors = {}
        for placement in placements:
            executor = StageExecutor.create(
                placement=placement,
                model_path=model_path,
                mock_mode=mock_mode,
                hidden_dim=config.hidden_dim,
                num_kv_heads=config.num_kv_heads,
            )
            executors[placement.stage_id] = executor

        pipeline = cls(session=session, executors=executors)

        # Initialize KV caches
        for executor in executors.values():
            executor.init_kv_cache(batch_size=1)

        return pipeline

    def _compute_input_hash(self, activation: np.ndarray) -> bytes:
        """Compute hash of input activation for receipt chaining."""
        return hashlib.sha256(activation.tobytes()).digest()[:16]

    def _create_receipt(
        self,
        executor: StageExecutor,
        token_index: int,
        stage_id: int,
        commitment: bytes,
        input_hash: bytes,
    ) -> Receipt:
        """Create a signed receipt for completed work."""
        signed = create_and_sign_receipt(
            key_manager=executor.key_manager,
            session_id=self.session.id,
            order_id=str(uuid.uuid4()),
            token_index=token_index,
            stage_id=stage_id,
            commitment=commitment,
            input_hash=input_hash,
            timestamp_ms=int(time.time() * 1000),
        )

        return Receipt(
            session_id=signed.data.session_id,
            order_id=signed.data.order_id,
            node_id=signed.data.node_id,
            token_index=signed.data.token_index,
            stage_id=signed.data.stage_id,
            commitment=signed.data.commitment,
            input_hash=signed.data.input_hash,
            timestamp_ms=signed.data.timestamp_ms,
            signature=signed.signature,
        )

    def execute_token(
        self,
        token_index: int,
        initial_embedding: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline for a single token.

        Args:
            token_index: Index of the token being generated
            initial_embedding: Input embedding (or None for mock mode)

        Returns:
            PipelineResult with output logits and metadata
        """
        start_time = time.perf_counter()

        # Track results per stage
        commitments: Dict[int, bytes] = {}
        compute_times: Dict[int, int] = {}
        work_units: Dict[int, WorkUnit] = {}

        # Initial activation (mock: seeded random embedding for determinism)
        if initial_embedding is None:
            rng = np.random.default_rng(self.session.config.rng_seed + token_index)
            current_activation = rng.standard_normal(
                (1, 1, self.session.config.hidden_dim)
            ).astype(np.float32)
        else:
            current_activation = initial_embedding

        # Execute each stage in sequence
        for stage_id in sorted(self.executors.keys()):
            executor = self.executors[stage_id]

            # Compute input hash for receipt chaining
            input_hash = self._compute_input_hash(current_activation)

            # Execute stage
            output_activation, commitment, compute_time_us = executor.execute(
                input_activation=current_activation,
                position_offset=len(self.session.prompt_tokens) + token_index,
            )

            # Store results
            commitments[stage_id] = commitment
            compute_times[stage_id] = compute_time_us

            # Create work unit
            order_id = str(uuid.uuid4())
            work_unit = WorkUnit(
                order_id=order_id,
                token_index=token_index,
                stage_id=stage_id,
                input_activation=current_activation,
                output_activation=output_activation,
                commitment=commitment,
                compute_time_us=compute_time_us,
            )

            # Create and store receipt
            receipt = self._create_receipt(
                executor=executor,
                token_index=token_index,
                stage_id=stage_id,
                commitment=commitment,
                input_hash=input_hash,
            )
            work_unit.receipt = receipt
            self.session.add_receipt(receipt)

            # Maybe audit this work unit
            if self.session.should_audit():
                audit_passed = self._verify_work(work_unit)
                work_unit.audited = True
                work_unit.audit_passed = audit_passed

                if self.on_audit_callback:
                    self.on_audit_callback(token_index, stage_id, audit_passed)

                if not audit_passed:
                    logger.warning(
                        f"Audit FAILED for token {token_index}, stage {stage_id}"
                    )

            work_units[stage_id] = work_unit

            # Pass output to next stage
            current_activation = output_activation

        # Final output is logits (in real model, would go through lm_head)
        output_logits = current_activation

        total_time_ms = (time.perf_counter() - start_time) * 1000

        # Store token generation record
        token_gen = TokenGeneration(
            token_index=token_index,
            work_units=work_units,
            final_logits=output_logits,
            total_time_ms=total_time_ms,
        )
        self.session.token_generations.append(token_gen)

        return PipelineResult(
            token_index=token_index,
            output_logits=output_logits,
            commitments=commitments,
            compute_times_us=compute_times,
            total_time_ms=total_time_ms,
        )

    def _verify_work(self, work_unit: WorkUnit) -> bool:
        """
        Verify a work unit by recomputing and checking commitment.

        In the full system, this would be done by a different node.
        Here we simulate by recomputing with the same executor.
        """
        executor = self.executors[work_unit.stage_id]

        # Recompute forward pass
        recomputed_output, recomputed_commitment, _ = executor.execute(
            input_activation=work_unit.input_activation,
            position_offset=len(self.session.prompt_tokens) + work_unit.token_index,
        )

        # Compare commitments
        return recomputed_commitment == work_unit.commitment

    def _sample_token(self, logits: np.ndarray) -> int:
        """
        Sample next token from logits.

        Uses greedy sampling (argmax) for determinism in v1.
        """
        # For mock mode, logits are random - just sample from them
        if self.session.config.temperature == 0.0:
            # Greedy: take argmax
            # Flatten to 1D if needed
            flat_logits = logits.flatten()
            return int(np.argmax(flat_logits) % self.session.config.vocab_size)
        else:
            # Temperature sampling (not used in v1 greedy mode)
            flat_logits = logits.flatten()
            probs = np.exp(flat_logits / self.session.config.temperature)
            probs = probs / probs.sum()
            return int(self.session.rng.choice(len(probs), p=probs))

    def run(self, max_tokens: Optional[int] = None) -> List[int]:
        """
        Run the complete inference loop.

        Args:
            max_tokens: Override max tokens (uses session config if None)

        Returns:
            List of generated token IDs
        """
        max_tokens = max_tokens or self.session.config.max_tokens

        self.session.start()
        logger.info(f"Starting inference session {self.session.id}")

        try:
            for token_idx in range(max_tokens):
                # Execute pipeline for this token
                result = self.execute_token(token_idx)

                # Sample next token
                next_token = self._sample_token(result.output_logits)
                self.session.generated_tokens.append(next_token)

                # Update token generation record
                self.session.token_generations[token_idx].sampled_token = next_token

                # Callback
                if self.on_token_callback:
                    self.on_token_callback(token_idx, next_token)

                logger.debug(
                    f"Token {token_idx}: {next_token} "
                    f"({result.total_time_ms:.1f}ms)"
                )

                # Check for EOS (in mock mode, we don't have real EOS)
                # In real mode, would check: if next_token == EOS_TOKEN_ID: break

            self.session.complete()
            logger.info(
                f"Session {self.session.id} completed: "
                f"{len(self.session.generated_tokens)} tokens in "
                f"{self.session.get_duration_ms():.1f}ms"
            )

        except Exception as e:
            self.session.fail(str(e))
            logger.error(f"Session {self.session.id} failed: {e}")
            raise

        return self.session.generated_tokens

    def get_summary(self) -> dict:
        """Get summary of pipeline execution."""
        audit_summary = self.session.get_audit_summary()

        return {
            "session_id": self.session.id,
            "status": self.session.status.value,
            "tokens_generated": len(self.session.generated_tokens),
            "duration_ms": self.session.get_duration_ms(),
            "tokens_per_second": self.session.get_tokens_per_second(),
            "receipts": len(self.session.receipts),
            "audits": audit_summary,
            "stages": [
                {
                    "stage_id": p.stage_id,
                    "layers": f"{p.layer_start}-{p.layer_end}",
                    "node_id": p.node_id,
                }
                for p in self.session.placements
            ],
        }
