# Consortium v1 Implementation Plan
## Heterogeneous Distributed Inference with Fuzzy Verification

**Version**: 1.0
**Target**: End-to-End Test Case v1
**Hardware**: MacBook Air M4 (24GB) + Windows Desktop (RTX 4070 Super)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture-v1)
3. [Canonical-Grid Commitment Spec](#3-canonical-grid-commitment-spec)
4. [Protocol & APIs](#4-protocol--apis)
5. [Core Algorithms](#5-core-algorithms)
6. [Implementation Plan](#6-implementation-plan)
7. [Test Plan](#7-test-plan)
8. [Runbook](#8-runbook)
9. [Risk Register](#9-risk-register--mitigations)
10. [Acceptance Criteria](#10-acceptance-criteria)

---

## 1. Executive Summary

### What Will Be Built

A proof-of-concept distributed inference system demonstrating:
- **Pipeline-parallel inference** across heterogeneous hardware (CUDA + Metal)
- **Fuzzy verification** using canonical-grid commitments that tolerate floating-point variance
- **Optimistic audits** with random sampling (20% audit rate)
- **Fault tolerance** with automatic failover when a node dies mid-generation

### What Will Be Proven

1. A model (Llama-3-8B) can be partitioned across devices with different compute backends
2. Canonical-grid hashing prevents false fraud proofs despite numerical differences
3. Random audits correctly validate honest computation
4. The system recovers gracefully from node failure without corrupting output

### Definition of Done

Test Case v1 passes when:
- 64 tokens are generated for a fixed prompt across 3 logical nodes
- Output matches single-node baseline (token-for-token with deterministic sampling)
- All random audits (≈13 of 64×3 = 192 work units) pass verification
- Forced node failure at token 20 triggers successful failover
- No session abort, no false fraud detection, generation completes

---

## 2. System Architecture (v1)

### 2.1 Logical Node Configuration

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LOGICAL TOPOLOGY                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │    NODE 0   │     │    NODE 1   │     │    NODE 2   │          │
│   │ Coordinator │     │   Worker    │     │   Worker    │          │
│   │ + Stage 2   │     │   Stage 0   │     │   Stage 1   │          │
│   │ + Verifier  │     │ Layers 0-10 │     │ Layers 11-21│          │
│   │ Layers 22-31│     │             │     │ + Verifier  │          │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘          │
│          │                   │                   │                  │
│          └───────────────────┴───────────────────┘                  │
│                              │                                      │
│                         gRPC/TCP                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Physical Deployment Options

**Option A: Full Simulation (Mac-only, Development)**
```
MacBook Air M4:
  - Process 0: Coordinator + Stage 2 (layers 22-31)
  - Process 1: Worker Stage 0 (layers 0-10)
  - Process 2: Worker Stage 1 (layers 11-21)

All processes communicate via localhost TCP.
```

**Option B: Two-Machine (Mac + Windows)**
```
Windows Desktop (RTX 4070 Super):
  - Process 0: Coordinator + Stage 2 (layers 22-31)
  - Process 1: Worker Stage 0 (layers 0-10)

MacBook Air M4:
  - Process 0: Worker Stage 1 (layers 11-21) + Verifier

Cross-machine communication via LAN TCP.
```

**Option C: Three-Machine Simulation (Windows with Docker)**
```
Windows Host:
  - Process: Coordinator + Stage 2

Windows Docker/WSL:
  - Container: Worker Stage 0

MacBook Air:
  - Process: Worker Stage 1 + Verifier
```

### 2.3 Component Responsibilities

| Component | Responsibilities |
|-----------|------------------|
| **Coordinator** | Session lifecycle, node registration, work dispatch, audit selection, failover orchestration, receipt collection |
| **Worker** | Load model shard, execute assigned layers, emit activations + commitments, respond to audits |
| **Verifier** | Recompute audited work, apply canonicalization, compare hashes, report results |

### 2.4 Data Flow for Single Token

```
Token Generation Flow (after prompt encoding):
═══════════════════════════════════════════════════════════════════

1. COORDINATOR dispatches WorkOrder{token_idx=N, prev_activation=h_prev}

2. STAGE 0 (Node 1):
   ├─ Receives h_prev (or input_ids if token_idx=0)
   ├─ Computes layers 0-10
   ├─ Produces h_10 (hidden state after layer 10)
   ├─ Generates commitment = canonical_hash(h_10)
   ├─ Returns WorkResult{activation=h_10, commitment=C0}
   └─ Stores Receipt{stage=0, token=N, commitment=C0, signature=sig}

3. STAGE 1 (Node 2):
   ├─ Receives h_10
   ├─ Computes layers 11-21
   ├─ Produces h_21
   ├─ Generates commitment = canonical_hash(h_21)
   ├─ Returns WorkResult{activation=h_21, commitment=C1}
   └─ Stores Receipt{stage=1, token=N, commitment=C1, signature=sig}

4. STAGE 2 (Node 0 / Coordinator):
   ├─ Receives h_21
   ├─ Computes layers 22-31 + final norm + lm_head
   ├─ Produces logits
   ├─ Samples next token
   ├─ Generates commitment = canonical_hash(logits)
   └─ Stores Receipt{stage=2, token=N, commitment=C2, signature=sig}

5. COORDINATOR:
   ├─ With probability 0.2, selects work unit for audit
   ├─ If auditing: dispatches AuditOrder to verifier
   └─ Appends token to sequence, loops to step 1
```

### 2.5 Receipt Storage

Receipts are stored locally by each node and collected by the coordinator:

```
receipts/
├── session_{session_id}/
│   ├── node_0/
│   │   ├── token_000_stage_2.json
│   │   ├── token_001_stage_2.json
│   │   └── ...
│   ├── node_1/
│   │   ├── token_000_stage_0.json
│   │   └── ...
│   └── node_2/
│       ├── token_000_stage_1.json
│       └── ...
```

---

## 3. Canonical-Grid Commitment Spec

### 3.1 Purpose

The canonical-grid commitment transforms floating-point tensors into a deterministic hash that:
- Is identical across CUDA and Metal backends (within tolerance)
- Enables verification without bit-exact floating-point reproducibility
- Is computationally cheap (applied only during verification, not inference)

### 3.2 Canonicalization Function (Exact Implementation)

```python
import numpy as np
import hashlib
import struct

def canonicalize_and_hash(tensor: np.ndarray) -> bytes:
    """
    Canonical-grid commitment as specified in architecture doc.

    Args:
        tensor: Activation tensor of any shape, float32 or float16

    Returns:
        32-byte SHA-256 hash (commitment)
    """
    # Step 1: Cast to float16
    y = tensor.astype(np.float16)

    # Step 2: Grid snap with factor 64
    # y_grid = round(y * 64) / 64
    y_grid = np.round(y * 64.0) / 64.0

    # Step 3: Clamp to [-100.0, 100.0]
    y_clamped = np.clip(y_grid, -100.0, 100.0)

    # Step 4: Serialize as little-endian bytes
    # Ensure contiguous memory layout in C order
    y_contiguous = np.ascontiguousarray(y_clamped)

    # Convert to bytes with explicit little-endian format
    # float16 is already 2 bytes, but we ensure byte order
    serialized = y_contiguous.tobytes()

    # Step 5: SHA-256 hash
    commitment = hashlib.sha256(serialized).digest()

    return commitment


def verify_commitment(tensor: np.ndarray, expected_hash: bytes) -> bool:
    """
    Verify that a tensor matches an expected commitment.

    Args:
        tensor: Recomputed activation tensor
        expected_hash: 32-byte commitment from original computation

    Returns:
        True if commitment matches, False otherwise
    """
    actual_hash = canonicalize_and_hash(tensor)
    return actual_hash == expected_hash
```

### 3.3 Parameters (Fixed for v1)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **dtype** | float16 | Reduces precision variance, matches inference dtype |
| **grid_factor** | 64 | ~0.0156 precision; empirically calibrated to absorb CUDA/Metal drift |
| **clamp_range** | [-100.0, 100.0] | Covers expected activation range; prevents outlier-driven hash instability |
| **hash_algo** | SHA-256 | Standard, fast, collision-resistant |
| **byte_order** | little-endian | x86/ARM native, explicit for cross-platform |

### 3.4 Where Canonicalization Is Applied

| Context | Apply Canonicalization? |
|---------|------------------------|
| Inference forward pass | **NO** - use full precision activations |
| Generating commitment hash | **YES** - worker applies before signing |
| Wire transfer of activations | **NO** - send full precision for accuracy |
| Audit verification | **YES** - verifier applies to recomputed output |

### 3.5 Grid Size Calibration Script

To empirically validate the grid factor of 64, run this calibration:

```python
# scripts/calibrate_grid.py

import numpy as np
from typing import List, Tuple

def calibrate_grid_factor(
    samples_cuda: List[np.ndarray],
    samples_metal: List[np.ndarray],
    grid_factors: List[int] = [16, 32, 64, 128, 256, 512]
) -> Tuple[int, float]:
    """
    Find minimum grid factor that achieves 100% hash match rate.

    Args:
        samples_cuda: Activation samples from CUDA backend
        samples_metal: Corresponding samples from Metal backend
        grid_factors: Factors to test

    Returns:
        (optimal_factor, match_rate)
    """
    results = []

    for factor in grid_factors:
        matches = 0
        total = len(samples_cuda)

        for cuda_act, metal_act in zip(samples_cuda, samples_metal):
            # Apply canonicalization with this factor
            cuda_grid = np.round(cuda_act.astype(np.float16) * factor) / factor
            metal_grid = np.round(metal_act.astype(np.float16) * factor) / factor

            cuda_grid = np.clip(cuda_grid, -100.0, 100.0)
            metal_grid = np.clip(metal_grid, -100.0, 100.0)

            cuda_hash = hashlib.sha256(cuda_grid.tobytes()).digest()
            metal_hash = hashlib.sha256(metal_grid.tobytes()).digest()

            if cuda_hash == metal_hash:
                matches += 1

        match_rate = matches / total
        results.append((factor, match_rate))
        print(f"Grid factor {factor}: {match_rate:.2%} match rate")

    # Find minimum factor with 100% match
    for factor, rate in results:
        if rate == 1.0:
            return factor, rate

    # Return best if none achieve 100%
    return max(results, key=lambda x: x[1])
```

---

## 4. Protocol & APIs

### 4.1 Message Schema Definitions

All messages use Protocol Buffers (protobuf) for cross-language compatibility.
Alternative: MessagePack for simpler Python-only implementation.

```protobuf
// consortium/proto/consortium.proto
syntax = "proto3";
package consortium.v1;

// ============ Node Registration ============

message NodeInfo {
    string node_id = 1;           // UUID
    string address = 2;           // "host:port"
    string hardware_desc = 3;     // "cuda:rtx4070" or "metal:m4"
    repeated string capabilities = 4;  // ["compute", "verify"]
    uint64 available_memory_mb = 5;
}

message RegisterNodeRequest {
    NodeInfo node = 1;
    bytes public_key = 2;         // Ed25519 public key for signing
}

message RegisterNodeResponse {
    bool accepted = 1;
    string error = 2;
    uint64 heartbeat_interval_ms = 3;
}

message HeartbeatRequest {
    string node_id = 1;
    uint64 timestamp_ms = 2;
    float load_factor = 3;        // 0.0-1.0 current utilization
}

message HeartbeatResponse {
    bool acknowledged = 1;
}

// ============ Session Management ============

message SessionConfig {
    string model_id = 1;          // "llama3-8b-q4_k_m"
    uint64 rng_seed = 2;          // Deterministic sampling seed
    float audit_probability = 3;   // 0.0-1.0
    uint32 max_tokens = 4;
    float temperature = 5;        // 0.0 for greedy
    uint32 top_k = 6;             // 1 for greedy
}

message StagePlacement {
    uint32 stage_id = 1;
    string node_id = 2;
    uint32 layer_start = 3;       // Inclusive
    uint32 layer_end = 4;         // Exclusive
}

message CreateSessionRequest {
    SessionConfig config = 1;
    repeated StagePlacement placements = 2;
    string prompt = 3;
}

message CreateSessionResponse {
    string session_id = 1;
    bool success = 2;
    string error = 3;
}

// ============ Work Orders ============

message WorkOrder {
    string session_id = 1;
    string order_id = 2;          // UUID for this specific work unit
    uint32 token_index = 3;
    uint32 stage_id = 4;
    bytes input_activation = 5;   // Serialized tensor (empty for stage 0, token 0)
    repeated uint32 input_token_ids = 6;  // For stage 0 initial encoding
    uint64 deadline_ms = 7;       // Absolute timestamp
}

message WorkResult {
    string order_id = 1;
    bytes output_activation = 2;  // Serialized tensor
    bytes commitment = 3;         // 32-byte SHA-256 hash
    uint64 compute_time_us = 4;
    bool success = 5;
    string error = 6;
}

// ============ Receipts ============

message Receipt {
    string session_id = 1;
    string order_id = 2;
    string node_id = 3;
    uint32 token_index = 4;
    uint32 stage_id = 5;
    bytes commitment = 6;
    bytes input_hash = 7;         // Hash of input activation (for chaining)
    uint64 timestamp_ms = 8;
    bytes signature = 9;          // Ed25519 signature over fields 1-8
}

// ============ Audits ============

message AuditOrder {
    string session_id = 1;
    string audit_id = 2;
    string target_order_id = 3;   // WorkOrder being audited
    string verifier_node_id = 4;
    bytes input_activation = 5;   // Input to recompute
    bytes expected_commitment = 6;
}

message AuditResult {
    string audit_id = 1;
    bool passed = 2;
    bytes computed_commitment = 3;
    string failure_reason = 4;    // If passed=false
    uint64 verify_time_us = 5;
}

// ============ Failure Handling ============

message NodeFailure {
    string failed_node_id = 1;
    uint32 failed_stage_id = 2;
    string last_successful_order_id = 3;
    uint64 detected_at_ms = 4;
}

message FailoverOrder {
    string session_id = 1;
    string backup_node_id = 2;
    uint32 stage_id = 3;
    uint32 layer_start = 4;
    uint32 layer_end = 5;
    bytes last_input_activation = 6;  // Resume from here
    uint32 resume_token_index = 7;
}

// ============ Service Definition ============

service CoordinatorService {
    rpc RegisterNode(RegisterNodeRequest) returns (RegisterNodeResponse);
    rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
    rpc SubmitWorkResult(WorkResult) returns (Receipt);
    rpc SubmitAuditResult(AuditResult) returns (Receipt);
}

service WorkerService {
    rpc ExecuteWork(WorkOrder) returns (WorkResult);
    rpc ExecuteAudit(AuditOrder) returns (AuditResult);
    rpc AcceptFailover(FailoverOrder) returns (WorkResult);
    rpc GetStatus(Empty) returns (NodeInfo);
}
```

### 4.2 Activation Tensor Serialization

```python
# consortium/serialization.py

import struct
import numpy as np
from dataclasses import dataclass
from typing import Tuple

MAGIC = b'CACT'  # Consortium ACTivation
VERSION = 1

@dataclass
class ActivationHeader:
    version: int
    dtype: int          # 0=float16, 1=float32
    ndim: int
    shape: Tuple[int, ...]

    def to_bytes(self) -> bytes:
        # Magic + version + dtype + ndim + shape dims
        header = struct.pack('<4sHBB', MAGIC, self.version, self.dtype, self.ndim)
        for dim in self.shape:
            header += struct.pack('<Q', dim)
        return header

    @classmethod
    def from_bytes(cls, data: bytes) -> Tuple['ActivationHeader', int]:
        magic, version, dtype, ndim = struct.unpack('<4sHBB', data[:8])
        assert magic == MAGIC, f"Invalid magic: {magic}"

        offset = 8
        shape = []
        for _ in range(ndim):
            dim, = struct.unpack('<Q', data[offset:offset+8])
            shape.append(dim)
            offset += 8

        return cls(version=version, dtype=dtype, ndim=ndim, shape=tuple(shape)), offset


def serialize_activation(tensor: np.ndarray) -> bytes:
    """Serialize activation tensor for network transfer."""
    # Ensure float16 for transfer
    tensor = tensor.astype(np.float16)
    tensor = np.ascontiguousarray(tensor)

    header = ActivationHeader(
        version=VERSION,
        dtype=0,  # float16
        ndim=len(tensor.shape),
        shape=tensor.shape
    )

    return header.to_bytes() + tensor.tobytes()


def deserialize_activation(data: bytes) -> np.ndarray:
    """Deserialize activation tensor from network."""
    header, offset = ActivationHeader.from_bytes(data)

    dtype = np.float16 if header.dtype == 0 else np.float32
    tensor = np.frombuffer(data[offset:], dtype=dtype)
    tensor = tensor.reshape(header.shape)

    return tensor.copy()  # Ensure writable
```

### 4.3 Signing and Identity

Minimal viable identity for v1 (Ed25519 signatures):

```python
# consortium/identity.py

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
import hashlib

class NodeIdentity:
    def __init__(self, private_key: Ed25519PrivateKey = None):
        self.private_key = private_key or Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        self.node_id = self._derive_node_id()

    def _derive_node_id(self) -> str:
        """Derive node ID from public key."""
        pub_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        return hashlib.sha256(pub_bytes).hexdigest()[:16]

    def sign(self, data: bytes) -> bytes:
        """Sign data with private key."""
        return self.private_key.sign(data)

    def verify(self, signature: bytes, data: bytes) -> bool:
        """Verify signature."""
        try:
            self.public_key.verify(signature, data)
            return True
        except Exception:
            return False

    def sign_receipt(self, receipt_data: dict) -> bytes:
        """Sign receipt fields."""
        # Deterministic serialization
        canonical = (
            receipt_data['session_id'].encode() +
            receipt_data['order_id'].encode() +
            receipt_data['node_id'].encode() +
            struct.pack('<II', receipt_data['token_index'], receipt_data['stage_id']) +
            receipt_data['commitment'] +
            receipt_data['input_hash'] +
            struct.pack('<Q', receipt_data['timestamp_ms'])
        )
        return self.sign(canonical)
```

### 4.4 Versioning Strategy

- Protocol version in message header (currently v1)
- Breaking changes increment major version
- Nodes reject incompatible versions during registration
- Model configuration includes llama.cpp commit hash for reproducibility

---

## 5. Core Algorithms

### 5.1 Coordinator Session Setup + Placement

```python
# pseudocode: coordinator.setup_session()

def setup_session(config: SessionConfig, prompt: str) -> Session:
    """
    Initialize inference session with node placement.
    """
    # 1. Validate registered nodes
    available_nodes = [n for n in registered_nodes if n.is_healthy()]
    assert len(available_nodes) >= 3, "Need at least 3 nodes for 3-stage pipeline"

    # 2. Assign stages to nodes
    # Strategy: Coordinator takes final stage (has sampling logic)
    # Balance layers roughly equally
    total_layers = 32  # Llama-3-8B
    layers_per_stage = [11, 11, 10]  # 0-10, 11-21, 22-31

    placements = [
        StagePlacement(stage_id=0, node_id=nodes[1].id, layer_start=0,  layer_end=11),
        StagePlacement(stage_id=1, node_id=nodes[2].id, layer_start=11, layer_end=22),
        StagePlacement(stage_id=2, node_id=nodes[0].id, layer_start=22, layer_end=32),
    ]

    # 3. Create session state
    session = Session(
        id=generate_uuid(),
        config=config,
        placements=placements,
        prompt_tokens=tokenize(prompt),
        generated_tokens=[],
        receipts=[],
        rng=RandomState(config.rng_seed),
    )

    # 4. Notify workers to load model shards
    for placement in placements:
        node = get_node(placement.node_id)
        await node.prepare_shard(
            model_id=config.model_id,
            layer_start=placement.layer_start,
            layer_end=placement.layer_end,
        )

    # 5. Initialize KV caches on all nodes
    for placement in placements:
        node = get_node(placement.node_id)
        await node.init_kv_cache(session.id, max_seq_len=config.max_tokens + len(session.prompt_tokens))

    return session
```

### 5.2 Token Loop Orchestration

```python
# pseudocode: coordinator.run_inference()

async def run_inference(session: Session) -> List[int]:
    """
    Main token generation loop with pipeline execution.
    """
    # Encode prompt through pipeline (no sampling)
    prompt_activations = await encode_prompt(session)

    # Generation loop
    for token_idx in range(session.config.max_tokens):
        # Track activation flowing through pipeline
        current_activation = None

        for stage_idx, placement in enumerate(session.placements):
            # Prepare work order
            order = WorkOrder(
                session_id=session.id,
                order_id=generate_uuid(),
                token_index=token_idx,
                stage_id=stage_idx,
                input_activation=serialize(current_activation) if current_activation else b'',
                deadline_ms=now_ms() + STAGE_TIMEOUT_MS,
            )

            # Dispatch to worker
            node = get_node(placement.node_id)
            try:
                result = await asyncio.wait_for(
                    node.execute_work(order),
                    timeout=STAGE_TIMEOUT_MS / 1000
                )
            except asyncio.TimeoutError:
                # Trigger failover
                result = await handle_stage_failure(session, stage_idx, order)

            # Validate and store receipt
            receipt = create_receipt(order, result, node)
            session.receipts.append(receipt)

            # Maybe audit this work unit
            if session.rng.random() < session.config.audit_probability:
                await schedule_audit(session, order, result)

            # Pass activation to next stage
            current_activation = deserialize(result.output_activation)

        # Stage 2 output is logits - sample next token
        logits = current_activation
        next_token = sample_token(logits, session.config, session.rng)
        session.generated_tokens.append(next_token)

        # Check for EOS
        if next_token == EOS_TOKEN_ID:
            break

    return session.generated_tokens
```

### 5.3 Audit Sampling and Recomputation

```python
# pseudocode: coordinator.audit_work()

async def schedule_audit(session: Session, order: WorkOrder, result: WorkResult):
    """
    Schedule audit for a completed work unit.
    """
    # Select verifier (different from original worker)
    original_node = get_node_for_stage(session, order.stage_id)
    verifier_candidates = [n for n in session.nodes if n.id != original_node.id and 'verify' in n.capabilities]
    verifier = random.choice(verifier_candidates)

    audit_order = AuditOrder(
        session_id=session.id,
        audit_id=generate_uuid(),
        target_order_id=order.order_id,
        verifier_node_id=verifier.id,
        input_activation=order.input_activation,
        expected_commitment=result.commitment,
    )

    # Execute audit asynchronously
    asyncio.create_task(execute_audit(session, audit_order, verifier))


async def execute_audit(session: Session, audit_order: AuditOrder, verifier: Node):
    """
    Execute audit on verifier node.
    """
    try:
        audit_result = await verifier.execute_audit(audit_order)

        if not audit_result.passed:
            # FRAUD DETECTED - log and potentially slash
            log_fraud(session, audit_order, audit_result)
            # For v1: just log, don't abort session
        else:
            log_audit_passed(session, audit_order, audit_result)

    except Exception as e:
        log_audit_error(session, audit_order, e)


# On verifier node:
def execute_audit_local(order: AuditOrder, model_shard) -> AuditResult:
    """
    Recompute work and verify commitment.
    """
    # Deserialize input
    input_activation = deserialize(order.input_activation)

    # Recompute forward pass for this stage
    output_activation = model_shard.forward(input_activation)

    # Apply canonicalization and hash
    computed_commitment = canonicalize_and_hash(output_activation)

    # Compare
    passed = (computed_commitment == order.expected_commitment)

    return AuditResult(
        audit_id=order.audit_id,
        passed=passed,
        computed_commitment=computed_commitment,
        failure_reason="" if passed else "Commitment mismatch",
    )
```

### 5.4 Failover Sequence

```python
# pseudocode: coordinator.handle_failure()

async def handle_stage_failure(session: Session, failed_stage: int, failed_order: WorkOrder) -> WorkResult:
    """
    Handle node failure mid-pipeline.

    Recovery strategy:
    1. Select backup node
    2. Send failover order with last known input
    3. Backup node loads required layers (if not already)
    4. Backup node recomputes from last input
    5. Resume pipeline
    """
    failed_placement = session.placements[failed_stage]
    failed_node_id = failed_placement.node_id

    log_failure(session, failed_stage, failed_node_id)

    # Find backup node (prefer coordinator for simplicity in v1)
    backup_node_id = select_backup_node(session, failed_stage)
    backup_node = get_node(backup_node_id)

    # Prepare failover order
    failover = FailoverOrder(
        session_id=session.id,
        backup_node_id=backup_node_id,
        stage_id=failed_stage,
        layer_start=failed_placement.layer_start,
        layer_end=failed_placement.layer_end,
        last_input_activation=failed_order.input_activation,
        resume_token_index=failed_order.token_index,
    )

    # Execute failover
    # This may require backup to:
    # 1. Load model layers if not present
    # 2. Rebuild KV cache from prompt (expensive but correct)
    result = await backup_node.accept_failover(failover)

    # Update session placement
    session.placements[failed_stage] = StagePlacement(
        stage_id=failed_stage,
        node_id=backup_node_id,
        layer_start=failed_placement.layer_start,
        layer_end=failed_placement.layer_end,
    )

    # Mark failed node as unhealthy
    mark_node_unhealthy(failed_node_id)

    return result


def select_backup_node(session: Session, failed_stage: int) -> str:
    """
    Select backup node for failover.

    Strategy for v1: Use coordinator (N0) as universal backup.
    Coordinator must have full model loaded or ability to load any shard.
    """
    coordinator_id = session.placements[-1].node_id  # Last stage is coordinator

    # Verify coordinator can handle this
    coordinator = get_node(coordinator_id)
    if coordinator.is_healthy() and coordinator.can_load_layers(
        session.placements[failed_stage].layer_start,
        session.placements[failed_stage].layer_end
    ):
        return coordinator_id

    # Fallback: find any healthy node with capacity
    for node in registered_nodes:
        if node.is_healthy() and node.id not in [p.node_id for p in session.placements]:
            return node.id

    raise NoBackupAvailableError(f"No backup available for stage {failed_stage}")
```

### 5.5 Receipt Scoring (v1: Simple Counters)

```python
# pseudocode: scoring.py

@dataclass
class NodeScore:
    node_id: str
    work_completed: int = 0
    work_failed: int = 0
    audits_passed: int = 0
    audits_failed: int = 0

    @property
    def reliability_score(self) -> float:
        total = self.work_completed + self.work_failed
        if total == 0:
            return 1.0
        return self.work_completed / total

    @property
    def honesty_score(self) -> float:
        total = self.audits_passed + self.audits_failed
        if total == 0:
            return 1.0
        return self.audits_passed / total


class ReceiptScorer:
    def __init__(self):
        self.scores: Dict[str, NodeScore] = {}

    def record_work_complete(self, node_id: str):
        self._ensure_node(node_id)
        self.scores[node_id].work_completed += 1

    def record_work_failed(self, node_id: str):
        self._ensure_node(node_id)
        self.scores[node_id].work_failed += 1

    def record_audit_passed(self, node_id: str):
        self._ensure_node(node_id)
        self.scores[node_id].audits_passed += 1

    def record_audit_failed(self, node_id: str):
        self._ensure_node(node_id)
        self.scores[node_id].audits_failed += 1

    def get_summary(self) -> Dict[str, dict]:
        return {
            node_id: {
                'work_completed': score.work_completed,
                'work_failed': score.work_failed,
                'audits_passed': score.audits_passed,
                'audits_failed': score.audits_failed,
                'reliability': score.reliability_score,
                'honesty': score.honesty_score,
            }
            for node_id, score in self.scores.items()
        }

    def _ensure_node(self, node_id: str):
        if node_id not in self.scores:
            self.scores[node_id] = NodeScore(node_id=node_id)
```

---

## 6. Implementation Plan

### 6.1 Directory Structure

```
consortium/
├── README.md
├── pyproject.toml              # Python project config
├── requirements.txt
├── requirements-dev.txt
│
├── src/
│   └── consortium/
│       ├── __init__.py
│       ├── config.py           # Configuration dataclasses
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── session.py      # Session state management
│       │   ├── placement.py    # Stage placement logic
│       │   └── tokenizer.py    # Tokenizer wrapper
│       │
│       ├── coordinator/
│       │   ├── __init__.py
│       │   ├── server.py       # gRPC server
│       │   ├── orchestrator.py # Token generation loop
│       │   ├── auditor.py      # Audit scheduling and execution
│       │   ├── failover.py     # Failure detection and recovery
│       │   └── scorer.py       # Receipt scoring
│       │
│       ├── worker/
│       │   ├── __init__.py
│       │   ├── server.py       # gRPC server
│       │   ├── executor.py     # Work execution logic
│       │   ├── model_shard.py  # Model loading and layer execution
│       │   └── kv_cache.py     # KV cache management
│       │
│       ├── verification/
│       │   ├── __init__.py
│       │   ├── canonicalizer.py    # Canonical grid implementation
│       │   ├── commitment.py       # Commitment generation
│       │   └── verifier.py         # Audit verification logic
│       │
│       ├── transport/
│       │   ├── __init__.py
│       │   ├── serialization.py    # Tensor serialization
│       │   ├── messages.py         # Message dataclasses
│       │   └── grpc_impl.py        # gRPC service implementations
│       │
│       ├── identity/
│       │   ├── __init__.py
│       │   ├── keys.py         # Key generation and management
│       │   └── signing.py      # Receipt signing
│       │
│       └── cli/
│           ├── __init__.py
│           ├── main.py         # Entry point
│           ├── coordinator.py  # Coordinator CLI
│           ├── worker.py       # Worker CLI
│           └── verifier.py     # Verifier CLI
│
├── proto/
│   └── consortium.proto        # Protocol buffer definitions
│
├── llama_cpp/                  # Vendored llama.cpp (submodule or copy)
│   ├── README.md
│   ├── CMakeLists.txt
│   └── ...                     # Full llama.cpp source
│
├── scripts/
│   ├── setup_model.py          # Download and convert model
│   ├── calibrate_grid.py       # Grid factor calibration
│   ├── benchmark.py            # Performance benchmarks
│   └── inject_fault.py         # Fault injection helper
│
├── tests/
│   ├── conftest.py
│   ├── test_canonicalizer.py
│   ├── test_serialization.py
│   ├── test_single_node.py     # Baseline single-node test
│   ├── test_distributed.py     # Multi-process distributed test
│   ├── test_failover.py        # Failover scenario test
│   └── integration/
│       ├── test_e2e_local.py   # Full E2E on Mac
│       └── test_e2e_mixed.py   # E2E with Mac + Windows
│
├── configs/
│   ├── default.yaml            # Default configuration
│   ├── mac_local.yaml          # Mac-only multi-process
│   ├── mac_windows.yaml        # Mixed Mac + Windows
│   └── docker_compose.yaml     # Docker setup for Windows simulation
│
└── docs/
    ├── architecture/           # Existing architecture docs
    ├── runbook.md              # Operational runbook
    └── api.md                  # API documentation
```

### 6.2 Key Classes and Files

| File | Class/Function | Responsibility |
|------|----------------|----------------|
| `core/session.py` | `Session` | Holds session state: placements, tokens, receipts |
| `coordinator/orchestrator.py` | `Orchestrator` | Main token generation loop, dispatches work orders |
| `coordinator/auditor.py` | `Auditor` | Selects work units for audit, executes verification |
| `coordinator/failover.py` | `FailoverManager` | Detects failures, selects backups, coordinates recovery |
| `worker/model_shard.py` | `ModelShard` | Loads specific layers, executes forward pass |
| `worker/kv_cache.py` | `KVCacheManager` | Manages per-node KV cache, supports checkpoint/restore |
| `verification/canonicalizer.py` | `canonicalize_and_hash()` | Implements canonical grid commitment |
| `transport/serialization.py` | `serialize_activation()` | Tensor serialization for network transfer |
| `identity/signing.py` | `NodeIdentity` | Key management and receipt signing |

### 6.3 Dependencies

```toml
# pyproject.toml
[project]
name = "consortium"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # Core
    "numpy>=1.24",
    "torch>=2.0",

    # Model loading
    "llama-cpp-python>=0.2.90",  # Pinned version TBD after testing
    "transformers>=4.40",        # For tokenizer
    "huggingface-hub",

    # Networking
    "grpcio>=1.60",
    "grpcio-tools>=1.60",
    "protobuf>=4.0",

    # Identity/Crypto
    "cryptography>=42.0",

    # Config
    "pyyaml",
    "pydantic>=2.0",

    # CLI
    "click>=8.0",
    "rich",  # Pretty terminal output
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio",
    "pytest-timeout",
    "mypy",
    "ruff",
]
```

### 6.4 CLI Commands

```bash
# Start coordinator
consortium coordinator start \
    --config configs/default.yaml \
    --host 0.0.0.0 \
    --port 50051 \
    --model llama3-8b-q4_k_m

# Start worker
consortium worker start \
    --config configs/default.yaml \
    --coordinator-address localhost:50051 \
    --host 0.0.0.0 \
    --port 50052 \
    --node-id worker-1 \
    --capabilities compute,verify

# Run inference session
consortium session run \
    --coordinator-address localhost:50051 \
    --prompt "Explain in one paragraph why the sky appears blue." \
    --max-tokens 64 \
    --audit-probability 0.2 \
    --seed 42

# Check session status
consortium session status --session-id <id>

# Run baseline comparison
consortium test baseline \
    --model llama3-8b-q4_k_m \
    --prompt "Explain in one paragraph why the sky appears blue." \
    --max-tokens 64 \
    --seed 42

# Inject fault (for testing)
consortium test inject-fault \
    --target-node worker-2 \
    --at-token 20 \
    --fault-type kill

# Run full test suite
consortium test e2e \
    --config configs/mac_local.yaml
```

---

## 7. Test Plan

### 7.1 Baseline Single-Node Run

**Purpose**: Establish ground truth output for comparison.

```python
# tests/test_single_node.py

def test_single_node_baseline():
    """
    Run complete inference on single node.
    This output is the reference for distributed runs.
    """
    model = load_full_model("llama3-8b-q4_k_m")

    prompt = "Explain in one paragraph why the sky appears blue."

    output = model.generate(
        prompt=prompt,
        max_tokens=64,
        temperature=0.0,
        top_k=1,
        seed=42,
    )

    # Save baseline
    save_baseline(output, "baseline_64tokens.json")

    # Verify determinism
    output2 = model.generate(
        prompt=prompt,
        max_tokens=64,
        temperature=0.0,
        top_k=1,
        seed=42,
    )

    assert output.tokens == output2.tokens, "Single-node run must be deterministic"
```

### 7.2 Distributed Run Test

**Purpose**: Verify distributed inference matches baseline.

```python
# tests/test_distributed.py

@pytest.fixture
def distributed_cluster():
    """Start 3-node cluster for testing."""
    # Start coordinator
    coord = start_coordinator(port=50051)

    # Start workers
    worker1 = start_worker(port=50052, layers=(0, 11))
    worker2 = start_worker(port=50053, layers=(11, 22))

    yield {
        'coordinator': coord,
        'workers': [worker1, worker2],
    }

    # Cleanup
    coord.stop()
    worker1.stop()
    worker2.stop()


def test_distributed_matches_baseline(distributed_cluster):
    """
    Distributed inference must produce same tokens as single-node.
    """
    baseline = load_baseline("baseline_64tokens.json")

    result = run_distributed_session(
        coordinator=distributed_cluster['coordinator'],
        prompt="Explain in one paragraph why the sky appears blue.",
        max_tokens=64,
        seed=42,
    )

    assert result.tokens == baseline.tokens, (
        f"Token mismatch at positions: "
        f"{[i for i, (a, b) in enumerate(zip(result.tokens, baseline.tokens)) if a != b]}"
    )
```

### 7.3 Token-by-Token Diff Checks

```python
# tests/test_distributed.py

def test_token_by_token_comparison(distributed_cluster):
    """
    Compare each token with baseline, capture activation diffs.
    """
    baseline = load_baseline("baseline_64tokens.json")

    result = run_distributed_session_with_traces(
        coordinator=distributed_cluster['coordinator'],
        prompt="Explain in one paragraph why the sky appears blue.",
        max_tokens=64,
        seed=42,
        capture_activations=True,
    )

    report = []
    for i, (baseline_tok, result_tok) in enumerate(zip(baseline.tokens, result.tokens)):
        match = baseline_tok == result_tok
        report.append({
            'index': i,
            'baseline': baseline_tok,
            'result': result_tok,
            'match': match,
            'stage_timings': result.stage_timings[i],
        })

        if not match:
            # Log detailed info for debugging
            print(f"MISMATCH at token {i}:")
            print(f"  Baseline: {baseline_tok} ({tokenizer.decode([baseline_tok])})")
            print(f"  Result:   {result_tok} ({tokenizer.decode([result_tok])})")

    # All must match
    assert all(r['match'] for r in report)
```

### 7.4 Audit Pass Rate Expectations

```python
# tests/test_audits.py

def test_audit_pass_rate(distributed_cluster):
    """
    All audits must pass for honest computation.
    Expected: 100% pass rate (0 false positives).
    """
    result = run_distributed_session(
        coordinator=distributed_cluster['coordinator'],
        prompt="Explain in one paragraph why the sky appears blue.",
        max_tokens=64,
        seed=42,
        audit_probability=0.2,
    )

    # Expect ~38 audits (192 work units * 0.2)
    audits = result.audit_results
    assert len(audits) > 0, "Expected some audits to occur"

    passed = [a for a in audits if a.passed]
    failed = [a for a in audits if not a.passed]

    # Critical: NO false positives allowed
    assert len(failed) == 0, (
        f"False fraud detected! {len(failed)} audits failed:\n" +
        "\n".join(f"  - {a.audit_id}: {a.failure_reason}" for a in failed)
    )

    print(f"Audit summary: {len(passed)}/{len(audits)} passed (100%)")
```

### 7.5 Fault Injection Procedure

```python
# tests/test_failover.py

def test_fault_injection_at_token_20(distributed_cluster):
    """
    Kill worker-2 at token 20, verify recovery.
    """
    # Start session
    session = start_session(
        coordinator=distributed_cluster['coordinator'],
        prompt="Explain in one paragraph why the sky appears blue.",
        max_tokens=64,
        seed=42,
    )

    # Register fault injection callback
    def on_token_generated(token_idx):
        if token_idx == 20:
            # Kill worker-2 (stage 1: layers 11-21)
            distributed_cluster['workers'][1].kill()
            print(f"FAULT INJECTED: Killed worker-2 at token {token_idx}")

    session.on_token_callback = on_token_generated

    # Run to completion
    result = session.run()

    # Verify recovery
    assert result.completed, "Session must complete despite failure"
    assert len(result.tokens) == 64, "Must generate all 64 tokens"
    assert result.failover_events, "Failover should have occurred"

    # Check failover event details
    failover = result.failover_events[0]
    assert failover.failed_stage == 1
    assert failover.resume_token == 20
    assert failover.backup_node == "coordinator"  # or other backup

    # Verify no false fraud during failover
    assert all(a.passed for a in result.audit_results), "No false fraud during failover"


def test_latency_spike_during_failover(distributed_cluster):
    """
    Measure latency impact of failover.
    """
    # Run with fault injection
    result = run_with_fault_at_token(distributed_cluster, fault_token=20)

    # Get per-token latencies
    latencies = result.per_token_latencies

    # Token 20 should have higher latency (failover + KV rebuild)
    normal_latency = statistics.mean(latencies[:19])
    failover_latency = latencies[20]

    print(f"Normal latency: {normal_latency:.1f}ms")
    print(f"Failover latency: {failover_latency:.1f}ms")
    print(f"Spike factor: {failover_latency / normal_latency:.1f}x")

    # Document but don't fail - latency spike is expected
    assert result.completed
```

### 7.6 Logs/Metrics to Capture

| Metric | Description | Where Logged |
|--------|-------------|--------------|
| `tokens_generated` | Count of tokens produced | Coordinator |
| `stage_latency_ms` | Time per stage execution | Each worker |
| `activation_size_bytes` | Size of transferred activations | Transport layer |
| `commitment_time_us` | Time to compute commitment hash | Canonicalizer |
| `audit_count` | Number of audits executed | Auditor |
| `audit_pass_rate` | Fraction of audits that passed | Auditor |
| `failover_count` | Number of failover events | FailoverManager |
| `failover_latency_ms` | Time to complete failover | FailoverManager |
| `kv_rebuild_time_ms` | Time to rebuild KV cache | KVCacheManager |
| `memory_usage_mb` | Peak memory per node | Each process |

---

## 8. Runbook

### 8.1 Environment Setup

#### Prerequisites (All Platforms)

```bash
# Python 3.10+
python --version  # Should be 3.10+

# Clone repository
git clone <repo-url> consortium
cd consortium

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Generate protobuf files
python -m grpc_tools.protoc \
    -I proto \
    --python_out=src/consortium/transport \
    --grpc_python_out=src/consortium/transport \
    proto/consortium.proto
```

#### Mac-Specific Setup

```bash
# Install llama.cpp with Metal support
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Verify Metal is working
python -c "from llama_cpp import Llama; print('Metal support:', Llama.supports_gpu_offload())"
```

#### Windows-Specific Setup

```bash
# Install CUDA toolkit 12.x first (from NVIDIA website)

# Install llama.cpp with CUDA support
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Verify CUDA is working
python -c "from llama_cpp import Llama; print('CUDA support:', Llama.supports_gpu_offload())"
```

### 8.2 Model Setup

```bash
# Download and convert model (requires HuggingFace account with Llama access)
python scripts/setup_model.py \
    --model-id meta-llama/Llama-3-8B-Instruct \
    --output-dir models/ \
    --quantization q4_k_m

# Verify model
ls -la models/llama-3-8b-instruct-q4_k_m.gguf
```

### 8.3 Mac-Only Multi-Process Simulation

```bash
# Terminal 1: Start Coordinator (also runs Stage 2)
consortium coordinator start \
    --config configs/mac_local.yaml \
    --port 50051 \
    --model models/llama-3-8b-instruct-q4_k_m.gguf \
    --layers 22-32 \
    --log-level debug

# Terminal 2: Start Worker 1 (Stage 0)
consortium worker start \
    --config configs/mac_local.yaml \
    --coordinator localhost:50051 \
    --port 50052 \
    --node-id worker-stage0 \
    --model models/llama-3-8b-instruct-q4_k_m.gguf \
    --layers 0-11

# Terminal 3: Start Worker 2 (Stage 1)
consortium worker start \
    --config configs/mac_local.yaml \
    --coordinator localhost:50051 \
    --port 50053 \
    --node-id worker-stage1 \
    --model models/llama-3-8b-instruct-q4_k_m.gguf \
    --layers 11-22

# Terminal 4: Run inference
consortium session run \
    --coordinator localhost:50051 \
    --prompt "Explain in one paragraph why the sky appears blue." \
    --max-tokens 64 \
    --seed 42 \
    --audit-probability 0.2

# Expected output:
# Session ID: abc123
# Generated 64 tokens in 12.3s
# Audits: 38 passed, 0 failed
# Output: "The sky appears blue due to a phenomenon called Rayleigh scattering..."
```

### 8.4 Mixed Mac + Windows over LAN

```bash
# === On Windows Desktop ===

# Find IP address
ipconfig  # Note the LAN IP, e.g., 192.168.1.100

# Terminal 1: Start Coordinator + Stage 2
consortium coordinator start \
    --config configs/mac_windows.yaml \
    --host 0.0.0.0 \
    --port 50051 \
    --model models/llama-3-8b-instruct-q4_k_m.gguf \
    --layers 22-32

# Terminal 2: Start Worker Stage 0
consortium worker start \
    --config configs/mac_windows.yaml \
    --coordinator 192.168.1.100:50051 \
    --host 0.0.0.0 \
    --port 50052 \
    --node-id windows-stage0 \
    --model models/llama-3-8b-instruct-q4_k_m.gguf \
    --layers 0-11

# === On MacBook ===

# Terminal 1: Start Worker Stage 1 + Verifier
consortium worker start \
    --config configs/mac_windows.yaml \
    --coordinator 192.168.1.100:50051 \
    --host 0.0.0.0 \
    --port 50052 \
    --node-id mac-stage1 \
    --model models/llama-3-8b-instruct-q4_k_m.gguf \
    --layers 11-22 \
    --capabilities compute,verify

# === Run Session (from either machine) ===

consortium session run \
    --coordinator 192.168.1.100:50051 \
    --prompt "Explain in one paragraph why the sky appears blue." \
    --max-tokens 64 \
    --seed 42 \
    --audit-probability 0.2
```

### 8.5 Fault Injection Test

```bash
# Start cluster as above, then:

# In a separate terminal, run session with fault injection
consortium test inject-fault \
    --coordinator localhost:50051 \
    --prompt "Explain in one paragraph why the sky appears blue." \
    --max-tokens 64 \
    --seed 42 \
    --kill-node worker-stage1 \
    --at-token 20

# Expected behavior:
# - Tokens 0-19 generated normally
# - Token 20: worker-stage1 killed, timeout detected
# - Coordinator initiates failover to backup
# - Tokens 21-63 generated with backup node
# - Session completes successfully

# Check logs for failover details
grep "FAILOVER" logs/coordinator.log
```

---

## 9. Risk Register + Mitigations

### 9.1 Non-Determinism and Tolerance

| Risk | Impact | Likelihood | Mitigation | Detection |
|------|--------|------------|------------|-----------|
| CUDA/Metal produce different floating-point results | False fraud proofs | High | Canonical-grid commitment with calibrated tolerance | Run cross-platform calibration script |
| Thread scheduling affects reduction order | Non-reproducible hashes | Medium | Force single-threaded execution, fixed RNG seed | Compare multiple runs on same hardware |
| cuBLAS Tensor Core non-determinism | Hash variance between runs | Medium | Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Baseline reproducibility test |

**Recommended Grid Factor**: Start with 64 (as in spec), run calibration with ~100 samples across both backends, increase if needed.

### 9.2 Activation Transfer Overhead

| Risk | Impact | Likelihood | Mitigation | Detection |
|------|--------|------------|------------|-----------|
| Large activation tensors slow inference | High latency | Medium | Use float16 for transfer, consider compression | Measure transfer time vs compute time |
| Network bandwidth bottleneck | Pipeline stalls | Low (LAN) | Batch multiple positions if needed | Monitor network utilization |

**Activation Size**: For Llama-3-8B, hidden state is `[1, seq_len, 4096]` = 8KB/token in float16. At 64 tokens, ~512KB transferred per stage.

### 9.3 KV-Cache Mismatch During Failover

| Risk | Impact | Likelihood | Mitigation | Detection |
|------|--------|------------|------------|-----------|
| Backup node has no KV cache for failed layers | Must rebuild from scratch | Certain on failover | Accept performance hit; rebuild KV from prompt | Measure failover latency |
| KV cache format differs between backends | Incorrect attention | Low | Serialize KV cache in canonical format | Validate output after failover |

**Failover Strategy**: Accept the performance cost of rebuilding KV cache. For 20 tokens of context and 11 layers, expect ~500ms additional latency.

### 9.4 Serialization Mismatch

| Risk | Impact | Likelihood | Mitigation | Detection |
|------|--------|------------|------------|-----------|
| Different tensor memory layouts | Corrupted data | Medium | Always use `np.ascontiguousarray()`, explicit dtype | Unit test serialization roundtrip |
| Endianness mismatch | Wrong values | Low (both x86 Little-endian) | Force little-endian in serialization | Cross-platform serialization test |
| Float16 inf/nan handling differs | Hash mismatch | Low | Clamp before hashing (already in spec) | Include inf/nan in test cases |

### 9.5 Additional Risks

| Risk | Impact | Likelihood | Mitigation | Detection |
|------|--------|------------|------------|-----------|
| llama.cpp doesn't expose layer-by-layer API | Blocks implementation | Medium | Use eval callback for activation extraction; may need to modify llama.cpp | Early prototype to validate API access |
| Model too large for Mac 24GB | OOM | Low (8B Q4 ~5GB) | Monitor memory, use memory mapping | Track memory in tests |
| gRPC adds unexpected latency | Slow token generation | Low | Profile transport layer, consider simpler TCP if needed | Benchmark with/without network |

---

## 10. Acceptance Criteria

### 10.1 Binary Pass/Fail Conditions

| Criterion | Pass Condition | Measurement |
|-----------|----------------|-------------|
| **Inference Completes** | Session generates all 64 tokens without abort | Session status = COMPLETED |
| **Output Correctness** | Generated tokens match single-node baseline exactly | `tokens_distributed == tokens_baseline` |
| **Audit Success** | 100% of audits pass (no false fraud detection) | `audit_failures == 0` |
| **Failover Success** | Session continues after node kill at token 20 | `failover_success == True`, generation continues |
| **Cross-Platform Verification** | Audits pass when Mac verifies Windows work and vice versa | `cross_platform_audits_passed == 100%` |

### 10.2 Quantitative Thresholds

| Metric | Acceptable Range | Notes |
|--------|------------------|-------|
| Total generation time | < 120 seconds | For 64 tokens, ~2 tokens/sec minimum |
| Per-token latency (normal) | < 2000 ms | Excluding first token (prompt encoding) |
| Failover latency | < 10000 ms | One-time cost for KV cache rebuild |
| Memory per node | < 12 GB | Leave headroom on Mac's 24GB |

### 10.3 Test Case v1 Sign-Off Checklist

```
[ ] Single-node baseline generates 64 tokens correctly
[ ] Single-node is deterministic (same tokens on repeated runs)
[ ] Mac multi-process cluster starts and nodes register
[ ] Distributed run produces same tokens as baseline
[ ] All audits pass (0 false positives)
[ ] Fault injection at token 20 triggers failover
[ ] Failover completes and generation continues
[ ] Post-failover tokens still match expected output
[ ] Cross-platform test (Mac + Windows) runs successfully
[ ] Cross-platform audits pass
[ ] All tests pass in CI
[ ] Runbook steps work as documented
```

---

## Appendix A: Explicit Assumptions

| Assumption | Value | Override |
|------------|-------|----------|
| Model | Llama-3-8B-Instruct | Change model_id in config |
| Quantization | Q4_K_M | GGUF file format |
| Total layers | 32 | Derived from model |
| llama.cpp version | TBD (will pin stable commit) | Set in requirements.txt |
| Python version | 3.10+ | pyproject.toml |
| Port range | 50051-50059 | Config file |
| Audit probability | 0.2 (20%) | Session config |
| Grid factor | 64 | canonicalizer.py |
| Default seed | 42 | CLI/config |
| Stage timeout | 30000 ms | Config |
| Max tokens | 64 | Session config |

---

## Appendix B: Development Order (Recommended)

### Phase 1: Foundation (Mac-only)
1. Set up project structure and dependencies
2. Implement canonicalizer and verify determinism
3. Build basic model shard loading with llama-cpp-python
4. Test activation extraction from single model

### Phase 2: Single-Node Pipeline
1. Implement work order/result messages
2. Build in-process pipeline (no networking)
3. Verify output matches non-pipelined inference
4. Add commitment generation

### Phase 3: Multi-Process (Mac Local)
1. Add gRPC transport layer
2. Implement coordinator and worker servers
3. Run 3-process test on Mac
4. Add audit logic

### Phase 4: Cross-Platform
1. Set up Windows environment
2. Run calibration script to verify grid factor
3. Test Mac-Windows mixed cluster
4. Verify cross-platform audits pass

### Phase 5: Fault Tolerance
1. Implement failure detection (timeouts)
2. Implement backup node selection
3. Implement KV cache rebuild
4. Test fault injection scenario

### Phase 6: Polish
1. CLI improvements
2. Logging and metrics
3. Documentation
4. CI/CD setup

---

*End of Implementation Plan v1*
