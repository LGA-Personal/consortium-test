# Consortium Roadmap: From PoC to Production

**Vision:** A blockchain-embedded distributed inference network with a ChatGPT-style interface where users can:

- **Spend cryptocurrency** to purchase compute for inference
- **Contribute compute** from their own machines to earn cryptocurrency
- **Exchange tokens** for USD or reinvest in more inference

This document outlines the complete journey from the current proof-of-concept to that production vision.

---

## Current State (v0.1)

The Consortium PoC is built on **exo**, a distributed LLM inference framework using **MLX** (Apple's Metal-accelerated ML library) for efficient inference on Apple Silicon. The Consortium verification layer adds cryptographic commitments and fuzzy verification on top.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CONSORTIUM STACK                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                     Consortium Layer                         │  │
│   │  • Canonical-grid commitments (SHA-256 of quantized tensors) │  │
│   │  • Fuzzy verification (tolerates FP variance across devices) │  │
│   │  • Optimistic audits (20% random sampling)                   │  │
│   │  • Coordinator orchestration + failover                      │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              ▲                                      │
│                              │                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                        exo Layer                             │  │
│   │  • Master/Worker architecture with election                  │  │
│   │  • Pipeline-parallel inference (model split by layers)       │  │
│   │  • libp2p-based peer discovery & routing                     │  │
│   │  • Shard downloading from HuggingFace                        │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              ▲                                      │
│                              │                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                        MLX Layer                             │  │
│   │  • Metal-accelerated inference on Apple Silicon              │  │
│   │  • Quantized models (4-bit, 8-bit) from mlx-community        │  │
│   │  • Efficient memory management for large models              │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Current Capabilities

| Capability                   | Status         | Notes                                         |
| ---------------------------- | -------------- | --------------------------------------------- |
| Pipeline-parallel inference  | ✅ Implemented | exo's Master/Worker with layer sharding       |
| Canonical-grid commitments   | ✅ Implemented | SHA-256 hash of float16-quantized activations |
| Fuzzy verification           | ✅ Implemented | Grid factor=64, clamp to [-100, 100]          |
| Optimistic audits            | ✅ Implemented | Configurable audit rate (default 20%)         |
| Fault tolerance              | ✅ Implemented | Election-based master failover                |
| Peer discovery               | ✅ Implemented | libp2p with mDNS for local networks           |
| Model support                | ✅ Implemented | Llama, DeepSeek, Qwen, Kimi via mlx-community |
| Cross-device (Mac → Mac)     | 🔶 Untested    | Same MLX backend, should work                 |
| Cross-platform (Mac + Linux) | 🔶 Untested    | Different backends, needs validation          |

### Key Components

| Component     | Path                               | Purpose                              |
| ------------- | ---------------------------------- | ------------------------------------ |
| Main entry    | `src/exo/main.py`                  | Node startup, election loop          |
| Master        | `src/exo/master/`                  | Orchestration, API, scheduling       |
| Worker        | `src/exo/worker/`                  | Model shard execution, KV cache      |
| Coordinator   | `src/exo/consortium/coordinator/`  | Session management, audits, failover |
| Verification  | `src/exo/consortium/verification/` | Canonical-grid commitments           |
| Identity      | `src/exo/consortium/identity/`     | Key management, signing              |
| Router        | `src/exo/routing/`                 | libp2p peer-to-peer networking       |
| Rust bindings | `rust/exo_pyo3_bindings/`          | Performance-critical networking      |

**What's missing for production:**

- Real multi-device testing validation
- Economic incentive layer
- Blockchain integration
- User-facing application (beyond current API)

---

## Phase 1: Core Validation

**Goal:** Prove the system works across real heterogeneous hardware before adding complexity.

**Duration:** 1-2 weeks

### 1.1 Cross-Device Testing (Mac → Mac)

**Objective:** Validate canonical-grid verification works across different Apple Silicon chips (M1, M2, M3, M4).

| Test                  | Description                       | Success Criteria                     |
| --------------------- | --------------------------------- | ------------------------------------ |
| Peer discovery        | Two Macs find each other via mDNS | Nodes appear in peer list            |
| Master election       | One node becomes master           | Election completes, single master    |
| Distributed inference | Split model across devices        | Correct output generated             |
| Verification match    | Canonical hashes match            | 0 false mismatches across 100 tokens |
| Fault tolerance       | Kill worker mid-generation        | Session recovers, output correct     |

**Deliverables:**

- [ ] Test harness script for multi-device runs
- [ ] Network configuration guide (ports, firewall)
- [ ] Results report with latency measurements

### 1.2 Cross-Platform Testing (Mac + Linux)

**Objective:** Validate interoperability between MLX (Metal) and MLX[CPU] or potential alternative backends.

> [!NOTE]
> exo currently uses MLX which is Apple-focused. Cross-platform support may require:
>
> - Using MLX[CPU] on Linux (slower but compatible)
> - Adding an alternative backend (llama.cpp, vLLM) for CUDA nodes
> - Ensuring canonical commitments match across backends

| Test                | Description                                    | Success Criteria                    |
| ------------------- | ---------------------------------------------- | ----------------------------------- |
| Mixed pipeline      | Mac handles some layers, Linux handles others  | Output matches single-node baseline |
| Canonical tolerance | Different backends produce same canonical hash | Hash match rate ≥ 99.9%             |
| Bidirectional audit | Mac audits Linux work, and vice versa          | No false fraud proofs               |

**Deliverables:**

- [ ] Linux setup guide (MLX[CPU] or alternative backend)
- [ ] Cross-platform test script
- [ ] Variance analysis report (where do differences occur?)

### 1.3 Performance Benchmarking

**Metrics to capture:**

- Tokens/second (distributed vs single-node)
- Per-stage latency breakdown
- Network bandwidth utilization (libp2p overhead)
- Memory usage per node (MLX allocations)

**Deliverables:**

- [ ] Benchmark suite
- [ ] Performance report with charts
- [ ] Identified bottlenecks and optimization opportunities

---

## Phase 2: Network Layer

**Goal:** Enable nodes to find each other and communicate across network boundaries.

**Duration:** 2-4 weeks

### 2.1 Extend libp2p Discovery

**Objective:** Leverage exo's existing libp2p stack for broader discovery.

> [!NOTE]
> exo already uses libp2p with mDNS for local discovery. This phase extends it for:
>
> - WAN connectivity (DHT-based discovery)
> - NAT traversal (AutoNAT, relay protocols)
> - Hole punching (DCUtR protocol)

```
┌─────────────────────────────────────────────────────────────┐
│                     libp2p Discovery                        │
├─────────────────────────────────────────────────────────────┤
│  Local Network (existing):                                  │
│    • mDNS for peer discovery                                │
│    • Direct TCP connections                                 │
│                                                             │
│  WAN Extension (new):                                       │
│    • Kademlia DHT for peer routing                          │
│    • AutoNAT for NAT detection                              │
│    • Circuit Relay for fallback connectivity                │
│    • DCUtR for hole punching                                │
└─────────────────────────────────────────────────────────────┘
```

**Deliverables:**

- [ ] Enable DHT discovery in libp2p config
- [ ] Add AutoNAT/Relay support
- [ ] Test connectivity across different network types

### 2.2 Node Registry & Reputation

**Objective:** Track node availability, performance, and reliability.

**Initial implementation (SQLite, local):**

```python
class NodeRecord:
    node_id: str          # libp2p peer ID (base58)
    public_key: bytes     # Ed25519 from identity module
    last_seen: datetime
    uptime_ratio: float   # 0.0 - 1.0
    avg_latency_ms: float
    successful_audits: int
    failed_audits: int
    compute_contributed: int  # tokens computed
```

**Deliverables:**

- [ ] Node registry database schema
- [ ] Heartbeat/ping system
- [ ] Reputation scoring algorithm

### 2.3 Desktop Application Shell

**Objective:** Package the node software for easy installation.

**Options:**

| Framework        | Pros                            | Cons                      |
| ---------------- | ------------------------------- | ------------------------- |
| **Tauri** (Rust) | Tiny binary, native performance | Newer, smaller ecosystem  |
| **Electron**     | Huge ecosystem, easy UI         | Large binary, memory hog  |
| **PyInstaller**  | Already Python, minimal work    | Not a real app experience |

**Recommendation:** Start with PyInstaller for fast iteration, migrate to Tauri for production.

**Deliverables:**

- [ ] Installable package for macOS
- [ ] System tray icon
- [ ] Simple status UI (node running, connected peers, tokens computed)

---

## Phase 3: Economics Layer

**Goal:** Implement the incentive mechanics without blockchain (fast iteration).

**Duration:** 3-4 weeks

### 3.1 Off-Chain Accounting

**Objective:** Track compute contributions and consumption without blockchain overhead.

```
┌──────────────────────────────────────────────────────────┐
│                    Accounting Ledger                     │
├──────────────────────────────────────────────────────────┤
│  account_id  │  balance  │  earned  │  spent  │  staked │
├──────────────┼───────────┼──────────┼─────────┼─────────┤
│  alice       │  1,500    │  2,000   │  500    │  1,000  │
│  bob         │  250      │  0       │  750    │  0      │
│  charlie     │  3,200    │  3,500   │  300    │  2,000  │
└──────────────────────────────────────────────────────────┘
```

**Design decisions:**

- 1 token = 1 token of inference (initially, price can float later)
- New users get small free allocation for trial
- Compute providers earn tokens proportional to work

**Deliverables:**

- [ ] Account management system
- [ ] Transaction logging (immutable append-only log)
- [ ] Balance checking API

### 3.2 Compute Marketplace

**Objective:** Match inference requests with available compute.

**Request flow:**

```
User submits prompt
        │
        ▼
┌───────────────┐
│   Scheduler   │──► Find available nodes with capacity
└───────────────┘
        │
        ▼
┌───────────────┐
│   Matcher     │──► Select nodes based on: latency, reputation, price
└───────────────┘
        │
        ▼
┌───────────────┐
│   Executor    │──► Run distributed inference, verify, settle
└───────────────┘
```

**Deliverables:**

- [ ] Job queue system
- [ ] Node selection algorithm
- [ ] Request/response API

### 3.3 Pricing Mechanism

**Initial approach: Fixed pricing**

- 1 token = 1 inference token generated
- Simple, predictable, easy to understand

**Future approach: Dynamic pricing**

- Price based on demand/supply
- Premium for faster response
- Discount for off-peak usage

**Deliverables:**

- [ ] Pricing engine
- [ ] Usage metering
- [ ] Invoice generation

### 3.4 Web Dashboard

**Objective:** User interface for managing account and viewing activity.

**Features:**

- View balance, earnings, spending
- Configure node settings
- View inference history
- Simple chat interface for testing

**Deliverables:**

- [ ] Dashboard UI (React/Vue/Svelte)
- [ ] API endpoints for dashboard
- [ ] Authentication system

---

## Phase 4: Blockchain Integration

**Goal:** Decentralize trust and enable real economic value.

**Duration:** 4-8 weeks

### 4.1 Chain Selection

**Considerations:**

| Chain                  | Pros                          | Cons                              |
| ---------------------- | ----------------------------- | --------------------------------- |
| **Solana**             | Fast, cheap, large ecosystem  | Complexity, occasional outages    |
| **Avalanche Subnet**   | Customizable, EVM-compatible  | Less ecosystem than Solana        |
| **Base/Optimism** (L2) | Ethereum security, lower fees | Still relatively expensive        |
| **Custom L1**          | Full control                  | Massive effort, bootstrap problem |

**Recommendation:** Start with **Solana** or **Avalanche Subnet** for balance of speed, cost, and ecosystem.

### 4.2 Smart Contract Architecture

**Core contracts:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Contract Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐                                       │
│  │   Token Contract │  ERC-20 / SPL token                   │
│  └────────┬─────────┘                                       │
│           │                                                  │
│  ┌────────▼─────────┐                                       │
│  │  Staking Contract│  Lock tokens to become compute node   │
│  └────────┬─────────┘                                       │
│           │                                                  │
│  ┌────────▼─────────┐                                       │
│  │ Registry Contract│  Node registration, capabilities      │
│  └────────┬─────────┘                                       │
│           │                                                  │
│  ┌────────▼─────────┐                                       │
│  │ Commitment Store │  Record canonical hashes on-chain     │
│  └────────┬─────────┘                                       │
│           │                                                  │
│  ┌────────▼─────────┐                                       │
│  │ Slashing Contract│  Penalize nodes that fail verification│
│  └──────────────────┘                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Deliverables:**

- [ ] Token contract (mint, transfer, burn)
- [ ] Staking contract (stake, unstake, slash)
- [ ] Commitment contract (submit hash, verify)
- [ ] Contract tests

### 4.3 Bridge: Off-Chain ↔ On-Chain

**Challenge:** Not every inference needs to go on-chain (too expensive, too slow).

**Solution: Batch settlement**

```
Off-chain: Record every transaction in local ledger
             │
             ▼ (every N minutes or M transactions)
On-chain:  Submit merkle root of batch
             │
             ▼
           Anyone can verify inclusion
           Anyone can dispute
```

**Deliverables:**

- [ ] Batch aggregation service
- [ ] Merkle tree construction
- [ ] Dispute resolution mechanism

### 4.4 Token Economics

**Token utility:**

1. **Payment** - Buy inference compute
2. **Staking** - Required to be a compute provider
3. **Governance** - Vote on protocol changes (later)

**Initial distribution (example):**

| Allocation        | Percentage | Purpose                     |
| ----------------- | ---------- | --------------------------- |
| Team              | 15%        | Incentive alignment         |
| Investors         | 20%        | Funding development         |
| Community rewards | 40%        | Compute provider incentives |
| Treasury          | 15%        | Future development          |
| Liquidity         | 10%        | DEX trading pairs           |

**Deliverables:**

- [ ] Tokenomics whitepaper
- [ ] Vesting schedule implementation
- [ ] Initial distribution plan

---

## Phase 5: Production

**Goal:** Ship to real users.

**Duration:** Ongoing

### 5.1 User-Facing Chat Application

**Objective:** ChatGPT-style interface anyone can use.

**Platforms:**

- Web app (primary)
- Mobile app (iOS, Android)
- Desktop app (for power users who also run nodes)

**Features:**

- Conversation history
- Multiple models
- Usage tracking
- Easy payment (credit card → tokens)

### 5.2 Fiat On/Off Ramps

**Objective:** Let users buy tokens with credit card, sell for USD.

**Options:**

- Partner with existing on-ramp (MoonPay, Transak)
- Direct credit card processing (Stripe + custody)

### 5.3 Security Hardening

**Requirements before mainnet:**

- Smart contract audit (Trail of Bits, OpenZeppelin)
- Penetration testing
- Bug bounty program
- Rate limiting and DDoS protection

### 5.4 Scaling

**Challenges at scale:**

- Coordinator becomes bottleneck → Decentralized coordination
- Single model → Model marketplace
- Fixed pricing → Dynamic market

---

## Risk Analysis

| Risk                                         | Likelihood | Impact   | Mitigation                                      |
| -------------------------------------------- | ---------- | -------- | ----------------------------------------------- |
| Canonical hashes don't match across hardware | Medium     | Critical | Extensive cross-device testing (Phase 1)        |
| NAT traversal doesn't work reliably          | High       | High     | libp2p relay fallback                           |
| Token has no value                           | Medium     | High     | Focus on utility before speculation             |
| Smart contract exploit                       | Medium     | Critical | Multiple audits, bug bounty                     |
| Regulatory issues                            | Medium     | High     | Legal consultation, jurisdiction selection      |
| No one contributes compute                   | Medium     | High     | Bootstrap with own machines, attractive rewards |

---

## Success Metrics

### Phase 1-2 (Technical)

- [ ] 100+ tokens generated across 2+ devices with 0 false fraud proofs
- [ ] Successful failover in < 5 seconds
- [ ] NAT traversal success rate > 80%

### Phase 3 (Economic)

- [ ] 10+ nodes contributing compute
- [ ] 100+ inference requests served
- [ ] Off-chain accounting accurate to the token

### Phase 4 (Blockchain)

- [ ] Smart contracts deployed to testnet
- [ ] 100+ on-chain settlements
- [ ] 0 critical vulnerabilities in audit

### Phase 5 (Production)

- [ ] 1,000+ registered users
- [ ] 100+ active compute providers
- [ ] Consistent token velocity

---

## Appendix: Technology Decisions

### Why exo + MLX?

- **Apple Silicon optimized**: MLX is purpose-built for Metal, achieving excellent performance on Mac
- **Distributed by design**: exo's Master/Worker architecture handles model sharding natively
- **libp2p networking**: Proven peer-to-peer networking with NAT traversal capabilities
- **Active development**: Both exo and MLX are actively maintained with growing communities
- **Quantization support**: 4-bit and 8-bit models from mlx-community enable large models on consumer hardware

### Why Rust for critical paths?

- Memory safety without GC
- Excellent performance
- Great async support (tokio)
- pyo3 bindings integrate cleanly with Python

### Why Python for orchestration?

- Rapid development
- Rich ML ecosystem (numpy for canonicalization)
- anyio for async coordination
- Easy to prototype economics layer

### Why not just use X?

| Alternative            | Why Not                                                        |
| ---------------------- | -------------------------------------------------------------- |
| llama.cpp              | Less integrated with exo; would need significant refactoring   |
| Together.ai, Replicate | Centralized, no compute contribution model                     |
| Bittensor              | Different architecture, complex integration, validator-focused |
| IPFS/Filecoin          | Storage, not compute                                           |
| Golem                  | General compute, not ML-optimized                              |

---

## Next Steps

**Immediate (this week):**

1. [ ] Set up cross-device test between two MacBooks
2. [ ] Document current network configuration
3. [ ] Run first real distributed inference

**Short-term (next 2 weeks):**

1. [ ] Complete Phase 1.1 cross-device testing
2. [ ] Capture baseline performance metrics
3. [ ] Document any issues found

---

_Document version: 1.1_
_Last updated: 2025-12-30_
