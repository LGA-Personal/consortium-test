# Phase 1: Core Validation - Report

**Date:** 2025-12-30  
**Status:** Complete  
**Scope:** Multi-Device Distributed Inference on macOS (Apple Silicon)

## Executive Summary

Phase 1 successfully validated the core capability of the `exo` (Consortium) system: executing large language models (LLMs) distributed across multiple consumer-grade MacBooks over a local Wi-Fi network. We confirmed peer discovery, automatic model sharding, and end-to-end inference using Llama-3.2-1B as the test model.

## 1. System Architecture

The system implements a **peer-to-peer distributed inference engine** designed for heterogeneous hardware.

### Logical Topology

- **Master Node:** Orchestrates the cluster. It hosts the OpenAI-compatible HTTP API, runs the `Election` logic, and calculates model placement (sharding).
- **Worker Nodes:** Dumb execution units. They join the cluster, download their assigned model shards, and execute tensor operations.
- **Discovery:** Uses **mDNS** (via `libp2p`) for zero-configuration peer discovery on local LANs.

### Data Plane vs. Control Plane

- **Control Plane (TCP/HTTP):**
  - Uses a custom Command/Event architecture (`src/exo/shared/types/commands.py`).
  - Master sends `PlaceInstance` commands; Workers reply with `RunnerStatusUpdated`.
- **Data Plane (MLX Ring):**
  - Uses `mlx.distributed` (Apple's MLX framework) with a **Ring** backend.
  - Nodes connect directly to each other via ephemeral TCP ports to pass tensor data (activations) during inference.

## 2. Key Design Design Decisions

### A. Stateless Startup

**Decision:** The system does not persist running instances across restarts.
**Rationale:** Simplifies recovery. If a node crashes, the cluster is "dirty." It is cleaner to restart the process and simply re-issue a `/place_instance` command than to try to reconcile a broken distributed state.
**Impact:** Users must run `/place_instance` explicitly after every system reboot before querying the API.

### B. Pipeline Parallelism (Default)

**Decision:** We use Pipeline Parallelism (splitting model layers `0-16` on Node A, `16-32` on Node B) rather than Tensor Parallelism by default for network efficiency.
**Rationale:** Consumer WiFi has high latency. Pipeline parallelism minimizes the number of synchronizations per token compared to Tensor Parallelism, which requires synchronization for every matrix multiplication.

### C. The "No-API" Worker

**Decision:** Worker nodes are started with `--no-api`.
**Rationale:**

1.  **Port Conflicts:** Prevents workers from fighting for port 52415.
2.  **Asset Dependency:** Avoids crashes related to missing web dashboard assets on headless/worker installs.
3.  **Role Clarity:** Workers strictly process shards; they do not handle user HTTP requests.

## 3. Key File References

| Component         | File                                      | Description                                                                   |
| :---------------- | :---------------------------------------- | :---------------------------------------------------------------------------- |
| **Orchestration** | `src/exo/main.py`                         | Entry point. Sets up the `Node` composition (Master, Worker, Router).         |
| **API**           | `src/exo/master/api.py`                   | FastAPI implementation. Handles `/place_instance` and `/v1/chat/completions`. |
| **Worker Logic**  | `src/exo/worker/main.py`                  | Orchestrates the download and lifecycle of a worker.                          |
| **MLX Backend**   | `src/exo/worker/runner/runner.py`         | The actual inference loop. Handles the `mlx` state machine.                   |
| **Connectivity**  | `src/exo/worker/engines/mlx/utils_mlx.py` | Handles `mlx.distributed.init` and Ring formation.                            |

## 4. Verification Results

- **Hardware:** 2x MacBook Air (Apple Silicon).
- **Model:** `mlx-community/Llama-3.2-1B-Instruct-4bit`.
- **Test Case:**
  1.  Master (A) and Worker (B) discovery via logs.
  2.  `POST /place_instance` successfully calculated sharding (50/50 split).
  3.  `POST /v1/chat/completions` returned coherent text response.
  4.  Verified `No instance found` behavior correctly triggers on restart (validating stateless design).

## 5. Next Steps (Phase 2)

- **Fuzzy Verification:** Implement the "consortium" logic to police the workers.
- **Resilience:** Handle worker dropouts more gracefully.
- **Automation:** Create a script to auto-place instances on startup.
