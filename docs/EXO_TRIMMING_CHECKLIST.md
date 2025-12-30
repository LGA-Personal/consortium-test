# EXO Vendor Directory Trimming Checklist

## Overview

This document tracks what can be removed from `vendor/exo/` for the **Proof of Concept (POC)** vs what should be kept for the **Long-Term Vision** (blockchain-based compute marketplace with user apps).

**Long-Term Vision Summary:**
- Blockchain network for compute access
- Users buy compute with USD or earned crypto
- Users contribute compute and earn crypto rewards
- Native app (iOS/macOS) or web UI for chatbot interface
- Decentralized peer-to-peer infrastructure

---

## Key Finding: EXO is a Complete Distributed Systems Framework

After thorough examination, EXO contains **production-ready implementations** of:

| Component | Location | Blockchain Relevance |
|-----------|----------|---------------------|
| **Leader Election** | `src/exo/shared/election.py` | Consensus mechanism (clock-based voting) |
| **P2P Networking** | `rust/networking/` + bindings | libp2p (used by IPFS, Filecoin, Polkadot) |
| **GossipSub** | `src/exo/routing/router.py` | Message propagation (blockchain-style) |
| **Ed25519 Identity** | `src/exo/routing/router.py` | Cryptographic node identity |
| **Network Topology** | `src/exo/shared/topology.py` | Graph-based peer management |
| **Event Sourcing** | `src/exo/master/` | Event log (like blockchain ledger) |
| **Intelligent Placement** | `src/exo/master/placement.py` | Resource-aware job scheduling |

**The Rust code IS actively used** - `exo_pyo3_bindings` is imported by `routing/router.py` for NetworkingHandle, Keypair, and GossipSub operations.

---

## Examination Status

| Directory/File | Examined | POC Status | Long-Term Status | Notes |
|----------------|----------|------------|------------------|-------|
| `src/exo/` | YES | **KEEP** | **KEEP** | Core Python - inference + distributed systems |
| `rust/` | YES | **KEEP** | **KEEP** | libp2p networking - USED by Python code |
| `dashboard/` | YES | DELETE | **KEEP** | Full chat UI (Svelte + D3 topology viz) |
| `app/` | YES | DELETE | **KEEP** | macOS native app (Swift/SwiftUI) |
| `docs/` | YES | DELETE | **KEEP** | Architecture docs explain event sourcing |
| `.github/` | YES | DELETE | DELETE | CI/CD workflows |
| `.githooks/` | YES | DELETE | DELETE | Git hooks |
| `.idea/` | YES | DELETE | DELETE | JetBrains IDE config |
| `.vscode/` | YES | DELETE | DELETE | VS Code config |
| `.zed/` | YES | DELETE | DELETE | Zed editor config |
| `.mlx_typings/` | YES | OPTIONAL | OPTIONAL | Type stubs for IDE |
| `packaging/` | YES | DELETE | OPTIONAL | PyInstaller scripts |
| `tmp/` | YES | DELETE | DELETE | Temporary files |

---

## Detailed Component Analysis

### src/exo/ - Core Python (KEEP FOR BOTH)

#### src/exo/shared/ - Distributed Systems Primitives

| File | Purpose | Long-Term Relevance |
|------|---------|---------------------|
| `election.py` | **Distributed leader election** - clock-based voting, seniority, session management | **CRITICAL** - This is consensus |
| `topology.py` | **Network topology** - rustworkx graph, node/connection management, cycle detection | **CRITICAL** - P2P network structure |
| `apply.py` | State application logic | Event sourcing |
| `constants.py` | System constants | Configuration |

#### src/exo/master/ - Orchestration

| File | Purpose | Long-Term Relevance |
|------|---------|---------------------|
| `placement.py` | **Intelligent job placement** - finds optimal nodes by memory, topology, connection speed | **CRITICAL** - Resource scheduling |
| `placement_utils.py` | Placement helpers - shard assignment, cycle filtering | Job distribution |
| `api.py` | Master API | Control plane |
| `main.py` | Master entry point | Orchestration |

#### src/exo/routing/ - P2P Message Routing

| File | Purpose | Long-Term Relevance |
|------|---------|---------------------|
| `router.py` | **GossipSub routing** - uses Rust bindings, Ed25519 keypairs, topic-based pub/sub | **CRITICAL** - This is how blockchain nodes communicate |
| `topics.py` | Topic definitions | Message types |
| `connection_message.py` | Connection updates | Peer management |

#### src/exo/worker/ - Compute Execution

| File | Purpose | Long-Term Relevance |
|------|---------|---------------------|
| `runner/runner.py` | Inference execution (our modifications here) | Core inference |
| `engines/mlx/` | MLX pipeline parallelism (our modifications here) | Compute backend |

#### src/exo/utils/ - Utilities

| File | Purpose | Long-Term Relevance |
|------|---------|---------------------|
| `channels.py` | Async channels (tokio-style) | Message passing |
| `reactive.py` | Reactive state | UI updates |
| `event_buffer.py` | Event buffering | Event sourcing |

---

### rust/ - P2P Networking Stack (KEEP FOR BOTH)

**IMPORTANT:** This is NOT optional - it's actively imported by `src/exo/routing/router.py`

| Crate | Purpose | Long-Term Relevance |
|-------|---------|---------------------|
| `networking/` | **libp2p P2P networking** | **CRITICAL** - peer discovery, swarm, gossipsub |
| `exo_pyo3_bindings/` | **Python bindings** | **CRITICAL** - bridges Rust to Python |
| `system_custodian/` | System daemon - network config, IPv6 discovery | Network management |
| `util/` | Shared utilities | Required by above |

**Key Dependencies:**
- `libp2p` - Production P2P networking (IPFS, Filecoin, Polkadot use this)
- `tokio` - Async runtime
- `pyo3` - Rust-Python bindings
- `keccak-const` - Cryptographic hashing

**Files in `rust/networking/src/`:**
- `discovery.rs` - Peer discovery
- `swarm.rs` - Swarm networking
- `keep_alive.rs` - Connection keepalive

---

### dashboard/ - Web Chat UI (KEEP FOR LONG-TERM)

**Stack:** SvelteKit 5 + Tailwind 4 + D3.js

| Component | Size | Purpose |
|-----------|------|---------|
| `TopologyGraph.svelte` | 33KB | **D3 network visualization** - shows cluster topology |
| `ModelCard.svelte` | 29KB | Model selection/display |
| `ChatMessages.svelte` | 18KB | Chat message rendering |
| `ChatSidebar.svelte` | 17KB | Conversation list |
| `ChatForm.svelte` | 14KB | Chat input with attachments |
| `MarkdownContent.svelte` | 11KB | Markdown + LaTeX rendering |
| `ChatAttachments.svelte` | 2KB | File attachments |

**Dependencies:** marked (markdown), katex (LaTeX), highlight.js (code), d3 (graphs)

---

### app/ - macOS Native App (KEEP FOR LONG-TERM)

**Stack:** Swift + SwiftUI

| File | Purpose |
|------|---------|
| `ContentView.swift` (17KB) | Main app view |
| `InstanceRowView.swift` (12KB) | Instance list item |
| `EXOApp.swift` (8KB) | App entry point |
| `ExoProcessController.swift` (7KB) | Process lifecycle management |
| `TopologyMiniView.swift` (6KB) | Topology visualization |
| `ViewModels/*.swift` | MVVM view models |
| `Models/*.swift` | Data models |
| `Services/*.swift` | Backend services |

---

### docs/ - Architecture Documentation (KEEP FOR LONG-TERM)

| File | Content |
|------|---------|
| `architecture.md` | **Event sourcing architecture** - explains the Erlang-style message passing design |
| `imgs/` | Architecture diagrams |
| `benchmarks/` | Performance data |

**Key insight from architecture.md:**
> "EXO uses an Event Sourcing architecture, and Erlang-style message passing... Events are past tense, Commands are imperative."

This is exactly the pattern used in blockchain systems.

---

## Files We Modified (MUST KEEP)

These files contain our consortium integration hooks:

| File | Changes |
|------|---------|
| `src/exo/worker/engines/mlx/auto_parallel.py` | Added `verification_callback` parameter |
| `src/exo/worker/engines/mlx/utils_mlx.py` | Pass callback through loading chain |
| `src/exo/worker/runner/runner.py` | Initialize accumulator, emit events |
| `src/exo/shared/types/events.py` | Added `LayerCommitmentsGenerated` event |

---

## DELETE (Not Needed for Either)

| Item | Size | Reason |
|------|------|--------|
| `.github/` | 248KB | Use your own CI/CD |
| `.githooks/` | 4KB | Development tooling |
| `.idea/` | varies | JetBrains config |
| `.vscode/` | varies | VS Code config |
| `.zed/` | varies | Zed config |
| `tmp/` | 20KB | Temp files |
| `flake.nix` + `flake.lock` | 6KB | Nix (not used) |
| `justfile` | 1KB | Task runner |
| `.envrc` | 1KB | direnv |
| `.python-version` | 1KB | pyenv |
| `.gitignore` | 1KB | Use ours |
| `CONTRIBUTING.md` | 2KB | EXO dev guide |
| `PLATFORMS.md` | 1KB | Platform support |
| `RULES.md` | 5KB | Dev rules |
| `TODO.md` | 3KB | EXO todos |

---

## Root Files

| File | Size | POC | Long-Term | Purpose |
|------|------|-----|-----------|---------|
| `LICENSE` | 11KB | **KEEP** | **KEEP** | Legal requirement |
| `pyproject.toml` | 3KB | **KEEP** | **KEEP** | Package definition |
| `Cargo.toml` | 4KB | **KEEP** | **KEEP** | Rust workspace (needed for rust/) |
| `Cargo.lock` | 136KB | **KEEP** | **KEEP** | Rust dependencies |
| `uv.lock` | 290KB | DELETE | OPTIONAL | UV lock (regenerable) |
| `README.md` | 11KB | DELETE | OPTIONAL | EXO readme |
| `.clauderules` | 3KB | DELETE | OPTIONAL | AI rules reference |
| `.cursorrules` | 3KB | DELETE | OPTIONAL | AI rules reference |

---

## Recommendations

### For POC: Minimal Cleanup Only

```bash
cd vendor/exo

# Remove ONLY the junk (keeps everything useful)
rm -rf .github/ .githooks/ .idea/ .vscode/ .zed/ tmp/
rm -f flake.lock flake.nix justfile .envrc .python-version .gitignore
rm -f CONTRIBUTING.md PLATFORMS.md RULES.md TODO.md

# Optional: remove large regenerable file
rm -f uv.lock
```

**Size after:** ~3.5MB (down from ~4MB)

### For Long-Term: Keep Everything Except CI/IDE

The entire `src/exo/`, `rust/`, `dashboard/`, `app/`, and `docs/` are valuable:

| Component | Value for Blockchain Vision |
|-----------|----------------------------|
| Leader election | Consensus mechanism |
| GossipSub routing | Transaction propagation |
| Ed25519 identity | Cryptographic node identity |
| Network topology | Peer management |
| Intelligent placement | Job scheduling |
| Event sourcing | Audit log / ledger |
| Dashboard UI | User interface reference |
| Native app | Mobile/desktop reference |

---

## Verification After Cleanup

```bash
# Ensure EXO can still be imported
python -c "from exo.worker.engines.mlx.auto_parallel import pipeline_auto_parallel; print('OK')"

# Ensure Rust bindings work (requires Rust toolchain)
python -c "from exo.routing.router import Router; print('OK')"

# Run consortium tests
python -m pytest tests/exo_integration/ -v
```

---

## Summary

**Don't delete much.** EXO is not just an inference engine - it's a complete distributed systems framework with:
- Consensus (leader election)
- P2P networking (libp2p/GossipSub)
- Cryptographic identity (Ed25519)
- Event sourcing (blockchain-style ledger)
- Intelligent scheduling (resource-aware placement)
- Full UI implementations (web + native)

All of these are directly applicable to your blockchain compute marketplace vision.

---

## Changelog

| Date | Update |
|------|--------|
| 2025-12-29 | Complete re-examination: discovered election, topology, GossipSub, event sourcing |
| 2025-12-29 | Revised with two-column POC vs Long-Term analysis |
| 2025-12-29 | Initial checklist created |
