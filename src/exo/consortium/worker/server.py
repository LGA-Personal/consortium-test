"""
Worker gRPC Server

Workers are responsible for:
- Executing assigned layers of the model
- Generating commitments for their work
- Responding to audit requests
- Handling failover takeover
"""

import logging
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import grpc
import numpy as np

from exo.consortium.identity.keys import KeyManager
from exo.consortium.identity.signing import create_and_sign_receipt
from exo.consortium.transport import consortium_pb2, consortium_pb2_grpc
from exo.consortium.transport.serialization import deserialize_activation, serialize_activation
from exo.consortium.verification.canonicalizer import compute_commitment, verify_commitment
from exo.consortium.worker.kv_cache import KVCacheCheckpointer, StageKVCache
from exo.consortium.worker.model_shard import ModelShard, ShardConfig

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    """State managed by the worker."""

    node_id: str
    capabilities: List[str] = field(default_factory=lambda: ["compute", "verify"])
    key_manager: KeyManager = field(default_factory=KeyManager)
    model_shard: Optional[ModelShard] = None
    kv_caches: Dict[str, StageKVCache] = field(default_factory=dict)  # session_id -> cache
    checkpointers: Dict[str, KVCacheCheckpointer] = field(default_factory=dict)


class WorkerServicer(consortium_pb2_grpc.WorkerServiceServicer):
    """gRPC service implementation for the worker."""

    def __init__(self, state: WorkerState):
        self.state = state

    def ExecuteWork(
        self,
        request: consortium_pb2.WorkOrder,
        context: grpc.ServicerContext,
    ) -> consortium_pb2.WorkResult:
        """Execute a work order (forward pass through assigned layers)."""
        start_time = time.perf_counter()

        try:
            # Deserialize input activation
            if request.input_activation:
                input_activation = deserialize_activation(request.input_activation)
            else:
                # Stage 0: encode tokens to embeddings
                input_activation = self._encode_tokens(list(request.input_token_ids))

            # Execute forward pass
            if self.state.model_shard is None:
                raise RuntimeError("Model shard not loaded")

            result = self.state.model_shard.forward(
                input_activation,
                position_offset=request.token_index,
            )

            # Update KV cache
            session_id = request.session_id
            if session_id in self.state.kv_caches:
                # KV cache update would happen here in real implementation
                pass

            # Generate commitment
            commitment = compute_commitment(result.hidden_states)

            # Compute elapsed time
            elapsed_us = int((time.perf_counter() - start_time) * 1_000_000)

            logger.debug(
                f"Executed work {request.order_id}: "
                f"token={request.token_index}, stage={request.stage_id}, "
                f"time={elapsed_us}us"
            )

            return consortium_pb2.WorkResult(
                order_id=request.order_id,
                output_activation=serialize_activation(result.hidden_states),
                commitment=commitment,
                compute_time_us=elapsed_us,
                success=True,
            )

        except Exception as e:
            logger.error(f"Work execution failed: {e}")
            return consortium_pb2.WorkResult(
                order_id=request.order_id,
                success=False,
                error=str(e),
            )

    def ExecuteAudit(
        self,
        request: consortium_pb2.AuditOrder,
        context: grpc.ServicerContext,
    ) -> consortium_pb2.AuditResult:
        """Execute an audit (recompute and verify commitment)."""
        start_time = time.perf_counter()

        try:
            # Deserialize input
            input_activation = deserialize_activation(request.input_activation)

            # Recompute forward pass
            if self.state.model_shard is None:
                raise RuntimeError("Model shard not loaded for verification")

            result = self.state.model_shard.forward(input_activation)

            # Compute commitment
            computed_commitment = compute_commitment(result.hidden_states)

            # Compare with expected
            passed = computed_commitment == request.expected_commitment

            elapsed_us = int((time.perf_counter() - start_time) * 1_000_000)

            if passed:
                logger.info(f"Audit {request.audit_id} PASSED")
            else:
                logger.warning(
                    f"Audit {request.audit_id} FAILED: commitment mismatch"
                )

            return consortium_pb2.AuditResult(
                audit_id=request.audit_id,
                passed=passed,
                computed_commitment=computed_commitment,
                failure_reason="" if passed else "Commitment mismatch",
                verify_time_us=elapsed_us,
            )

        except Exception as e:
            logger.error(f"Audit execution failed: {e}")
            return consortium_pb2.AuditResult(
                audit_id=request.audit_id,
                passed=False,
                failure_reason=str(e),
            )

    def AcceptFailover(
        self,
        request: consortium_pb2.FailoverOrder,
        context: grpc.ServicerContext,
    ) -> consortium_pb2.WorkResult:
        """Accept failover responsibility for a failed stage."""
        logger.info(
            f"Accepting failover for stage {request.stage_id} "
            f"(layers {request.layer_start}-{request.layer_end})"
        )

        try:
            # Load the required layers if not already loaded
            if self.state.model_shard is None:
                # Would need to load model shard here
                logger.warning("Model shard not loaded for failover")

            # Restore KV cache from checkpoint if available
            session_id = request.session_id
            if session_id in self.state.checkpointers:
                checkpointer = self.state.checkpointers[session_id]
                restored = checkpointer.restore_from_checkpoint(request.resume_token_index)

                if restored:
                    pos, cache = restored
                    self.state.kv_caches[session_id] = cache
                    logger.info(f"Restored KV cache from checkpoint at position {pos}")

            # Execute the work
            input_activation = deserialize_activation(request.last_input_activation)
            result = self.state.model_shard.forward(input_activation)

            commitment = compute_commitment(result.hidden_states)

            return consortium_pb2.WorkResult(
                order_id=str(uuid.uuid4()),
                output_activation=serialize_activation(result.hidden_states),
                commitment=commitment,
                success=True,
            )

        except Exception as e:
            logger.error(f"Failover failed: {e}")
            return consortium_pb2.WorkResult(
                success=False,
                error=str(e),
            )

    def GetStatus(
        self,
        request: consortium_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> consortium_pb2.NodeInfo:
        """Get current worker status."""
        return consortium_pb2.NodeInfo(
            node_id=self.state.node_id,
            hardware_desc="unknown",
            capabilities=self.state.capabilities,
            available_memory_mb=0,  # Would calculate actual memory
        )

    def _encode_tokens(self, token_ids: List[int]) -> np.ndarray:
        """Encode tokens to initial embeddings (for stage 0)."""
        if self.state.model_shard is not None:
            return self.state.model_shard.encode_tokens(token_ids)
        else:
            # Mock: return random embeddings
            hidden_dim = 4096
            return np.random.randn(1, len(token_ids), hidden_dim).astype(np.float32)


class Worker:
    """
    Main worker class that manages the gRPC server and model execution.
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 50052,
        coordinator_address: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        max_workers: int = 10,
    ):
        self.host = host
        self.port = port
        self.coordinator_address = coordinator_address
        self.max_workers = max_workers

        # Create state
        self.state = WorkerState(
            node_id=node_id or str(uuid.uuid4())[:8],
            capabilities=capabilities or ["compute", "verify"],
        )

        self.server: Optional[grpc.Server] = None
        self._coordinator_stub: Optional[consortium_pb2_grpc.CoordinatorServiceStub] = None
        self._heartbeat_task = None

    @property
    def address(self) -> str:
        """Get worker address."""
        return f"{self.host}:{self.port}"

    @property
    def node_id(self) -> str:
        """Get worker node ID."""
        return self.state.node_id

    def load_model_shard(
        self,
        model_path: str,
        layer_start: int,
        layer_end: int,
        mock_mode: bool = False,
    ) -> None:
        """Load model shard for this worker."""
        config = ShardConfig(
            model_path=model_path,
            layer_start=layer_start,
            layer_end=layer_end,
            mock_mode=mock_mode,
        )

        self.state.model_shard = ModelShard(config)
        self.state.model_shard.load()

        logger.info(f"Loaded model shard: layers {layer_start}-{layer_end}")

    def start(self) -> None:
        """Start the worker server."""
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers)
        )

        servicer = WorkerServicer(self.state)
        consortium_pb2_grpc.add_WorkerServiceServicer_to_server(
            servicer, self.server
        )

        self.server.add_insecure_port(self.address)
        self.server.start()

        logger.info(f"Worker {self.node_id} started on {self.address}")

        # Register with coordinator if address provided
        if self.coordinator_address:
            self._register_with_coordinator()

    def stop(self, grace: float = 5.0) -> None:
        """Stop the worker server."""
        if self.server:
            self.server.stop(grace)
            logger.info(f"Worker {self.node_id} stopped")

    def wait_for_termination(self) -> None:
        """Block until server terminates."""
        if self.server:
            self.server.wait_for_termination()

    def _get_coordinator_stub(self) -> consortium_pb2_grpc.CoordinatorServiceStub:
        """Get or create coordinator stub."""
        if self._coordinator_stub is None:
            if not self.coordinator_address:
                raise RuntimeError("Coordinator address not set")

            channel = grpc.insecure_channel(self.coordinator_address)
            self._coordinator_stub = consortium_pb2_grpc.CoordinatorServiceStub(channel)

        return self._coordinator_stub

    def _register_with_coordinator(self) -> bool:
        """Register this worker with the coordinator."""
        try:
            stub = self._get_coordinator_stub()

            request = consortium_pb2.RegisterNodeRequest(
                node=consortium_pb2.NodeInfo(
                    node_id=self.node_id,
                    address=self.address,
                    hardware_desc=self._get_hardware_desc(),
                    capabilities=self.state.capabilities,
                    available_memory_mb=self._get_available_memory_mb(),
                ),
                public_key=self.state.key_manager.public_key_bytes,
            )

            response = stub.RegisterNode(request, timeout=10.0)

            if response.accepted:
                logger.info(f"Registered with coordinator at {self.coordinator_address}")
                return True
            else:
                logger.error(f"Registration rejected: {response.error}")
                return False

        except grpc.RpcError as e:
            logger.error(f"Failed to register with coordinator: {e}")
            return False

    def send_heartbeat(self) -> bool:
        """Send heartbeat to coordinator."""
        try:
            stub = self._get_coordinator_stub()

            request = consortium_pb2.HeartbeatRequest(
                node_id=self.node_id,
                timestamp_ms=int(time.time() * 1000),
                load_factor=self._get_load_factor(),
            )

            response = stub.Heartbeat(request, timeout=5.0)
            return response.acknowledged

        except grpc.RpcError as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False

    def _get_hardware_desc(self) -> str:
        """Get hardware description string."""
        import platform

        system = platform.system().lower()
        machine = platform.machine()

        if system == "darwin":
            return f"metal:{machine}"
        elif system == "linux" or system == "windows":
            # Would detect CUDA here
            return f"cuda:{machine}"
        else:
            return f"cpu:{machine}"

    def _get_available_memory_mb(self) -> int:
        """Get available memory in MB."""
        try:
            import psutil
            return int(psutil.virtual_memory().available / (1024 * 1024))
        except ImportError:
            return 0

    def _get_load_factor(self) -> float:
        """Get current load factor (0.0-1.0)."""
        try:
            import psutil
            return psutil.cpu_percent() / 100.0
        except ImportError:
            return 0.0


def run_worker(
    host: str = "0.0.0.0",
    port: int = 50052,
    coordinator_address: Optional[str] = None,
    model_path: Optional[str] = None,
    layer_start: int = 0,
    layer_end: int = 11,
    mock_mode: bool = True,
) -> None:
    """Run a worker server (blocking)."""
    worker = Worker(
        host=host,
        port=port,
        coordinator_address=coordinator_address,
    )

    # Load model shard
    worker.load_model_shard(
        model_path=model_path or "",
        layer_start=layer_start,
        layer_end=layer_end,
        mock_mode=mock_mode,
    )

    worker.start()

    try:
        worker.wait_for_termination()
    except KeyboardInterrupt:
        worker.stop()
