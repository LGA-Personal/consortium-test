# Consortium

**Heterogeneous Distributed Inference with Fuzzy Verification**

A proof-of-concept distributed LLM inference system demonstrating:

- **Pipeline-parallel inference** across heterogeneous hardware (CUDA + Metal)
- **Fuzzy verification** using canonical-grid commitments that tolerate floating-point variance
- **Optimistic audits** with random sampling (20% audit rate)
- **Fault tolerance** with automatic failover when a node dies mid-generation

## Overview

Consortium splits a large language model (Llama-3-8B) across multiple machines with different hardware backends. Each machine computes a subset of transformer layers and produces cryptographic commitments to its work. A coordinator orchestrates the pipeline, randomly audits work, and handles node failures.

### Key Features

- **Cross-platform verification**: CUDA and Metal produce slightly different floating-point results. The canonical-grid commitment scheme absorbs these differences.
- **Pipeline parallelism**: The model is split by layers (not tensors), enabling efficient streaming between stages.
- **Optimistic execution**: Work proceeds without waiting for verification; audits happen asynchronously.
- **Fault tolerance**: If a node fails mid-generation, a backup node takes over.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LOGICAL TOPOLOGY                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │    NODE 0   │     │    NODE 1   │     │    NODE 2   │          │
│   │ Coordinator │     │   Worker    │     │   Worker    │          │
│   │ + Stage 2   │     │   Stage 0   │     │   Stage 1   │          │
│   │ Layers 22-31│     │ Layers 0-10 │     │ Layers 11-21│          │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘          │
│          │                   │                   │                  │
│          └───────────────────┴───────────────────┘                  │
│                              │                                      │
│                         gRPC/TCP                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- For Mac: Metal support (M1/M2/M3/M4)
- For Windows: CUDA toolkit 12.x

### Setup

```bash
# Clone repository
git clone <repo-url> consortium
cd consortium

# Create virtual environment
python -m venv venv
source .venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Platform-Specific Setup

**macOS (Metal)**:
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Windows (CUDA)**:
```bash
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Quick Start

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_canonicalizer.py -v
```

### Single-Node Development

```bash
# Start coordinator
consortium coordinator start --port 50051 --model models/llama-3-8b.gguf

# Start workers (in separate terminals)
consortium worker start --coordinator localhost:50051 --port 50052 --layers 0-11
consortium worker start --coordinator localhost:50051 --port 50053 --layers 11-22

# Run inference
consortium session run \
    --coordinator localhost:50051 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 128
```

## Project Structure

```
consortium/
├── src/consortium/
│   ├── core/              # Session and placement management
│   ├── coordinator/       # Coordinator server and orchestration
│   ├── worker/            # Worker execution and model shards
│   ├── verification/      # Canonical-grid commitment system
│   ├── transport/         # Serialization and gRPC
│   ├── identity/          # Key management and signing
│   └── cli/               # Command-line interface
├── proto/                 # Protocol buffer definitions
├── tests/                 # Test suite
├── configs/               # Configuration files
└── docs/                  # Documentation
```

## Documentation

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Detailed technical plan
- [Architecture Spec](docs/architecture/) - Original test case specification

## License

MIT

## Status

**Work in Progress** - This is a proof-of-concept implementation for Test Case v1.
