"""
Coordinator gRPC Server

The coordinator is responsible for:
- Managing node registration and health
- Orchestrating inference sessions
- Dispatching work orders to workers
- Collecting receipts
- Scheduling audits
- Handling failover
"""

import asyncio
import logging
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import grpc
import numpy as np

from consortium.core.session import (
    Receipt,
    Session,
    SessionConfig,
    SessionStatus,
    StagePlacement,
    TokenGeneration,
    WorkUnit,
)
from consortium.identity.keys import KeyManager
from consortium.transport import consortium_pb2, consortium_pb2_grpc
from consortium.transport.serialization import deserialize_activation, serialize_activation
from consortium.verification.canonicalizer import verify_commitment

logger = logging.getLogger(__name__)


@dataclass
class RegisteredNode:
    """Information about a registered worker node."""

    node_id: str
    address: str
    hardware_desc: str
    capabilities: List[str]
    available_memory_mb: int
    public_key: bytes
    last_heartbeat_ms: int = 0
    is_healthy: bool = True
    load_factor: float = 0.0

    def update_heartbeat(self, load_factor: float = 0.0) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat_ms = int(time.time() * 1000)
        self.load_factor = load_factor
        self.is_healthy = True


@dataclass
class CoordinatorState:
    """State managed by the coordinator."""

    nodes: Dict[str, RegisteredNode] = field(default_factory=dict)
    sessions: Dict[str, Session] = field(default_factory=dict)
    pending_audits: Dict[str, dict] = field(default_factory=dict)  # audit_id -> audit info
    key_manager: KeyManager = field(default_factory=KeyManager)
    heartbeat_timeout_ms: int = 15000


class CoordinatorServicer(consortium_pb2_grpc.CoordinatorServiceServicer):
    """gRPC service implementation for the coordinator."""

    def __init__(self, state: CoordinatorState):
        self.state = state

    def RegisterNode(
        self,
        request: consortium_pb2.RegisterNodeRequest,
        context: grpc.ServicerContext,
    ) -> consortium_pb2.RegisterNodeResponse:
        """Register a new worker node."""
        node_info = request.node

        # Check if node already registered
        if node_info.node_id in self.state.nodes:
            logger.warning(f"Node {node_info.node_id} already registered, updating")

        # Create registered node
        registered = RegisteredNode(
            node_id=node_info.node_id,
            address=node_info.address,
            hardware_desc=node_info.hardware_desc,
            capabilities=list(node_info.capabilities),
            available_memory_mb=node_info.available_memory_mb,
            public_key=request.public_key,
        )
        registered.update_heartbeat()

        self.state.nodes[node_info.node_id] = registered

        logger.info(
            f"Registered node {node_info.node_id} at {node_info.address} "
            f"({node_info.hardware_desc})"
        )

        return consortium_pb2.RegisterNodeResponse(
            accepted=True,
            heartbeat_interval_ms=5000,
        )

    def Heartbeat(
        self,
        request: consortium_pb2.HeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> consortium_pb2.HeartbeatResponse:
        """Process heartbeat from worker node."""
        node_id = request.node_id

        if node_id not in self.state.nodes:
            logger.warning(f"Heartbeat from unregistered node {node_id}")
            return consortium_pb2.HeartbeatResponse(acknowledged=False)

        self.state.nodes[node_id].update_heartbeat(request.load_factor)

        return consortium_pb2.HeartbeatResponse(acknowledged=True)

    def SubmitWorkResult(
        self,
        request: consortium_pb2.WorkResult,
        context: grpc.ServicerContext,
    ) -> consortium_pb2.Receipt:
        """Receive work result from worker and return signed receipt."""
        # Find the session and work order
        # In a full implementation, we'd track pending work orders

        # Create receipt
        receipt = consortium_pb2.Receipt(
            session_id="",  # Would be filled from work order tracking
            order_id=request.order_id,
            node_id="",  # Would be filled from context
            token_index=0,
            stage_id=0,
            commitment=request.commitment,
            input_hash=b"",
            timestamp_ms=int(time.time() * 1000),
            signature=b"",  # Would be signed
        )

        return receipt

    def SubmitAuditResult(
        self,
        request: consortium_pb2.AuditResult,
        context: grpc.ServicerContext,
    ) -> consortium_pb2.Receipt:
        """Receive audit result from verifier."""
        audit_id = request.audit_id

        if audit_id in self.state.pending_audits:
            audit_info = self.state.pending_audits[audit_id]

            if request.passed:
                logger.info(f"Audit {audit_id} PASSED")
            else:
                logger.warning(
                    f"Audit {audit_id} FAILED: {request.failure_reason}"
                )

            del self.state.pending_audits[audit_id]

        # Return acknowledgment receipt
        return consortium_pb2.Receipt(
            session_id="",
            order_id=audit_id,
            timestamp_ms=int(time.time() * 1000),
        )


class Coordinator:
    """
    Main coordinator class that manages the gRPC server and orchestration.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
    ):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.state = CoordinatorState()
        self.server: Optional[grpc.Server] = None
        self._worker_stubs: Dict[str, consortium_pb2_grpc.WorkerServiceStub] = {}

    @property
    def address(self) -> str:
        """Get coordinator address."""
        return f"{self.host}:{self.port}"

    def start(self) -> None:
        """Start the coordinator server."""
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers)
        )

        servicer = CoordinatorServicer(self.state)
        consortium_pb2_grpc.add_CoordinatorServiceServicer_to_server(
            servicer, self.server
        )

        self.server.add_insecure_port(self.address)
        self.server.start()

        logger.info(f"Coordinator started on {self.address}")

    def stop(self, grace: float = 5.0) -> None:
        """Stop the coordinator server."""
        if self.server:
            self.server.stop(grace)
            logger.info("Coordinator stopped")

    def wait_for_termination(self) -> None:
        """Block until server terminates."""
        if self.server:
            self.server.wait_for_termination()

    def get_worker_stub(self, node_id: str) -> consortium_pb2_grpc.WorkerServiceStub:
        """Get or create a gRPC stub for a worker node."""
        if node_id not in self._worker_stubs:
            if node_id not in self.state.nodes:
                raise ValueError(f"Unknown node: {node_id}")

            node = self.state.nodes[node_id]
            channel = grpc.insecure_channel(node.address)
            self._worker_stubs[node_id] = consortium_pb2_grpc.WorkerServiceStub(channel)

        return self._worker_stubs[node_id]

    def get_healthy_nodes(self) -> List[RegisteredNode]:
        """Get list of healthy nodes."""
        now_ms = int(time.time() * 1000)
        healthy = []

        for node in self.state.nodes.values():
            # Check if heartbeat is recent
            if now_ms - node.last_heartbeat_ms < self.state.heartbeat_timeout_ms:
                node.is_healthy = True
                healthy.append(node)
            else:
                node.is_healthy = False

        return healthy

    def create_session(
        self,
        config: SessionConfig,
        prompt_tokens: List[int],
        placements: Optional[List[StagePlacement]] = None,
    ) -> Session:
        """Create a new inference session."""
        # Use provided placements or auto-assign
        if placements is None:
            placements = self._auto_assign_placements(config)

        session = Session.create(
            config=config,
            placements=placements,
            prompt_tokens=prompt_tokens,
        )

        self.state.sessions[session.id] = session
        logger.info(f"Created session {session.id}")

        return session

    def _auto_assign_placements(
        self,
        config: SessionConfig,
    ) -> List[StagePlacement]:
        """Automatically assign stages to available nodes."""
        healthy_nodes = self.get_healthy_nodes()

        if len(healthy_nodes) < 3:
            raise RuntimeError(
                f"Need at least 3 healthy nodes, have {len(healthy_nodes)}"
            )

        # Simple assignment: first 3 nodes get stages
        layers_per_stage = config.num_layers // 3
        extra = config.num_layers % 3

        placements = []
        layer_start = 0

        for i in range(3):
            layer_count = layers_per_stage + (1 if i < extra else 0)
            layer_end = layer_start + layer_count

            placements.append(
                StagePlacement(
                    stage_id=i,
                    node_id=healthy_nodes[i].node_id,
                    layer_start=layer_start,
                    layer_end=layer_end,
                )
            )

            layer_start = layer_end

        return placements

    async def dispatch_work(
        self,
        session: Session,
        token_index: int,
        stage_id: int,
        input_activation: np.ndarray,
    ) -> consortium_pb2.WorkResult:
        """Dispatch work to a worker node."""
        placement = session.get_placement_for_stage(stage_id)
        stub = self.get_worker_stub(placement.node_id)

        # Create work order
        order = consortium_pb2.WorkOrder(
            session_id=session.id,
            order_id=str(uuid.uuid4()),
            token_index=token_index,
            stage_id=stage_id,
            input_activation=serialize_activation(input_activation),
            deadline_ms=int(time.time() * 1000) + session.config.stage_timeout_ms,
        )

        # Send to worker
        try:
            result = stub.ExecuteWork(
                order,
                timeout=session.config.stage_timeout_ms / 1000,
            )
            return result

        except grpc.RpcError as e:
            logger.error(f"Work dispatch failed: {e}")
            raise

    async def schedule_audit(
        self,
        session: Session,
        work_order_id: str,
        stage_id: int,
        input_activation: np.ndarray,
        expected_commitment: bytes,
    ) -> str:
        """Schedule an audit for completed work."""
        # Select verifier (different from original worker)
        placement = session.get_placement_for_stage(stage_id)
        original_node = placement.node_id

        # Find a different healthy node with verify capability
        verifier = None
        for node in self.get_healthy_nodes():
            if node.node_id != original_node and "verify" in node.capabilities:
                verifier = node
                break

        if verifier is None:
            logger.warning("No verifier available for audit")
            return ""

        audit_id = str(uuid.uuid4())

        # Store pending audit
        self.state.pending_audits[audit_id] = {
            "session_id": session.id,
            "work_order_id": work_order_id,
            "stage_id": stage_id,
            "verifier_node": verifier.node_id,
            "expected_commitment": expected_commitment,
            "scheduled_at_ms": int(time.time() * 1000),
        }

        # Create and send audit order
        audit_order = consortium_pb2.AuditOrder(
            session_id=session.id,
            audit_id=audit_id,
            target_order_id=work_order_id,
            verifier_node_id=verifier.node_id,
            input_activation=serialize_activation(input_activation),
            expected_commitment=expected_commitment,
        )

        # Send to verifier asynchronously
        stub = self.get_worker_stub(verifier.node_id)
        try:
            stub.ExecuteAudit(audit_order, timeout=30.0)
        except grpc.RpcError as e:
            logger.error(f"Audit dispatch failed: {e}")

        return audit_id


def run_coordinator(
    host: str = "0.0.0.0",
    port: int = 50051,
) -> None:
    """Run the coordinator server (blocking)."""
    coordinator = Coordinator(host=host, port=port)
    coordinator.start()

    try:
        coordinator.wait_for_termination()
    except KeyboardInterrupt:
        coordinator.stop()
