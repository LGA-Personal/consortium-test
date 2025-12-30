# Multi-Device Testing Guide

This guide walks you through setting up and running Consortium across multiple MacBooks on a local network (WiFi/LAN).

---

## Overview

### What You'll Test

1. **Peer Discovery** - Nodes automatically find each other via mDNS
2. **Master Election** - One node becomes the coordinator
3. **Distributed Inference** - Model split across multiple devices
4. **Verification** - Canonical-grid commitments match across devices

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Local Network                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────────┐         mDNS Discovery        ┌───────────────┐│
│   │   MacBook A   │ ◄───────────────────────────► │   MacBook B   ││
│   │   (Master)    │                               │   (Worker)    ││
│   │               │         libp2p P2P            │               ││
│   │  Layers 0-15  │ ◄───────────────────────────► │  Layers 16-31 ││
│   │               │                               │               ││
│   │  API: :52415  │                               │               ││
│   └───────────────┘                               └───────────────┘│
│                                                                     │
│   User submits prompt to MacBook A's API                           │
│   → Inference splits across both machines                          │
│   → Commitments verified on canonicalized activations              │
│   → Response returned to user                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware

- **2+ MacBooks** with Apple Silicon (M1, M2, M3, M4)
- **Same WiFi network** (or connected via Ethernet to same LAN)
- **16GB+ RAM** recommended per device (8GB minimum for smaller models)

### Software

- **macOS** 13.0+ (Ventura or later recommended)
- **Python 3.13** (required by the project)
- **git** (for cloning the repo)
- **uv** (Python package manager) - will be installed in setup

---

## Step-by-Step Setup

### Machine A (Your Primary Mac)

You already have the repo set up. Just ensure it's running correctly:

```bash
# Navigate to project
cd /Users/lashby/Projects/consortium-test

# Activate environment
source .venv/bin/activate

# Verify installation
consortium --help
```

Expected output:

```
usage: EXO [-h] [-q] [-v] [-m] [--no-api] [--api-port API_PORT]
```

---

### Machine B (Second Mac) - Complete Setup

Run these commands on the second MacBook:

#### 1. Install Prerequisites

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.13 via Homebrew
brew install python@3.13

# Install Rust (needed to compile the libp2p networking bindings)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env  # Add cargo to PATH

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell to get everything in PATH
source ~/.zshrc  # or restart terminal
```

> [!IMPORTANT]
> Rust is required because the project includes `exo_pyo3_bindings` - Rust networking code compiled as a Python extension. The `uv sync` step will fail without Rust installed.

#### 2. Clone the Repository

```bash
# Clone to same location (or wherever you prefer)
mkdir -p ~/Projects
cd ~/Projects
git clone <your-repo-url> consortium-test
cd consortium-test
```

> [!NOTE]
> Replace `<your-repo-url>` with your actual repo URL. If it's a private repo, you'll need to set up SSH keys or use HTTPS with credentials.

#### 3. Create Virtual Environment

```bash
# Create venv with Python 3.13
uv venv --python 3.13

# Activate it
source .venv/bin/activate
```

#### 4. Install Dependencies

```bash
# This installs all Python dependencies + builds Rust bindings
uv sync
```

This step will:

- Install all Python packages from `pyproject.toml`
- Build the Rust networking bindings (`exo_pyo3_bindings`)
- Install MLX for Apple Silicon acceleration

> [!IMPORTANT]
> The first `uv sync` may take 5-10 minutes as it compiles Rust code.

#### 5. Verify Installation

```bash
# Test the consortium command
consortium --help

# Should see:
# usage: EXO [-h] [-q] [-v] [-m] [--no-api] [--api-port API_PORT]
```

---

## Running the Test

### Step 1: Start Machine A (Force Master)

On your **primary MacBook** (Machine A):

```bash
cd /Users/lashby/Projects/consortium-test
source .venv/bin/activate

# Start as forced master with verbose output
consortium --force-master -v
```

Expected output:

```
INFO     Starting EXO
INFO     Starting node 12D3KooW...  (your node ID will be different)
INFO     Node elected Master
```

The node will:

- Generate a unique node ID (stored in `~/.exo/node_id.keypair`)
- Start the libp2p networking layer
- Become the master due to `--force-master`
- Start the API server on port 52415

### Step 2: Start Machine B (Worker)

On **Machine B**:

```bash
cd ~/Projects/consortium-test
source .venv/bin/activate

# Start as worker (no --force-master)
consortium -v
```

Expected output:

```
INFO     Starting EXO
INFO     Starting node 12D3KooW...  (different ID than Machine A)
INFO     Node 12D3KooW... elected master  (shows Machine A's ID)
```

### Step 3: Verify Peer Discovery

Within a few seconds, you should see on **both machines**:

**Machine A:**

```
INFO     mDNS discovered a new peer: 12D3KooW... (Machine B's ID)
INFO     Connected to peer ...
```

**Machine B:**

```
INFO     mDNS discovered a new peer: 12D3KooW... (Machine A's ID)
INFO     Connected to peer ...
```

> [!TIP]
> If peers don't discover each other within 30 seconds, see the Troubleshooting section below.

### Step 4: Run Inference

From **any machine on the network** (or Machine A itself), send a request to the master's API:

```bash
# Replace with Machine A's IP if running from Machine B
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello! What is the capital of France?"}],
    "max_tokens": 50,
    "stream": false
  }'
```

> [!NOTE]
> The first request will download the model (~1GB for 1B model, larger for 8B). This may take a few minutes.

### Step 5: Verify Distribution

Watch the logs on both machines. You should see:

- **Machine A**: Logs about handling part of the inference
- **Machine B**: Logs about handling another part

---

## Test Cases

### Test 1: Basic Connectivity

| What to Check             | How                                | Expected Result                      |
| ------------------------- | ---------------------------------- | ------------------------------------ |
| Nodes discover each other | Watch logs for "mDNS discovered"   | Both nodes see each other within 30s |
| Connection established    | Watch logs for "Connected to peer" | Bidirectional connection             |
| Master elected            | Watch logs for "elected master"    | Only one master (Machine A)          |

### Test 2: Distributed Inference

| What to Check          | How               | Expected Result              |
| ---------------------- | ----------------- | ---------------------------- |
| Request succeeds       | Send curl request | Valid JSON response          |
| Both nodes participate | Watch logs        | Both show inference activity |
| Output is coherent     | Read the response | Sensible text (not garbage)  |

### Test 3: Fault Tolerance (Optional)

| What to Check         | How                        | Expected Result                           |
| --------------------- | -------------------------- | ----------------------------------------- |
| Worker death handling | Kill Machine B mid-request | Machine A takes over or errors gracefully |
| Reconnection          | Restart Machine B          | Re-discovers and reconnects               |

---

## Network Configuration

### Ports Used

| Port              | Protocol | Purpose                   |
| ----------------- | -------- | ------------------------- |
| 52415             | TCP      | HTTP API (only on master) |
| 5353              | UDP      | mDNS discovery            |
| Random high ports | TCP      | libp2p peer-to-peer       |

### Firewall Settings

macOS should prompt you to allow incoming connections when you first run `consortium`. Click **Allow**.

If you need to manually configure:

```bash
# Check if firewall is enabled
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# If needed, add exception for Python
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/.venv/bin/python3
```

### Finding IP Addresses

To find Machine A's IP for curl from Machine B:

```bash
# On Machine A
ipconfig getifaddr en0  # WiFi
# or
ipconfig getifaddr en1  # Ethernet
```

---

## Troubleshooting

### Peers Don't Discover Each Other

**Symptoms**: No "mDNS discovered" messages after 30 seconds

**Causes & Fixes**:

1. **Different networks**: Ensure both Macs are on the exact same WiFi network

   ```bash
   # Check network name on each Mac
   networksetup -getairportnetwork en0
   ```

2. **Firewall blocking mDNS**: Temporarily disable firewall to test

   ```bash
   sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
   ```

3. **VPN interfering**: Disconnect any VPN on both machines

4. **Router isolation**: Some routers have "client isolation" that blocks peer-to-peer. Check router settings or try Ethernet.

### Connection Drops

**Symptoms**: "Connection closed" messages, inconsistent behavior

**Causes & Fixes**:

1. **Weak WiFi signal**: Move closer to router or use Ethernet
2. **Sleep mode**: Disable sleep on both Macs during testing
   ```bash
   caffeinate -d  # Prevents display sleep (Ctrl+C to stop)
   ```

### Model Download Fails

**Symptoms**: Error downloading from HuggingFace

**Causes & Fixes**:

1. **Rate limited**: Wait and retry, or authenticate

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

2. **Disk space**: Models are large (1-10GB). Check free space.

### API Not Responding

**Symptoms**: curl to port 52415 times out

**Causes & Fixes**:

1. **Wrong IP**: Double-check Machine A's IP address
2. **API not enabled**: By default API spawns. Check if `--no-api` was accidentally passed
3. **Port conflict**: Try different port: `consortium --force-master --api-port 52416`

---

## What's Next

After successful testing:

1. **Capture metrics**: Run multiple inferences and note latency
2. **Test larger models**: Try 8B models split across both machines
3. **Add more devices**: Try with 3+ MacBooks
4. **Test fault tolerance**: Kill workers and verify recovery

---

## Quick Reference

### Machine A Commands

```bash
cd /Users/lashby/Projects/consortium-test
source .venv/bin/activate
consortium --force-master -v
```

### Machine B Commands

```bash
cd ~/Projects/consortium-test
source .venv/bin/activate
consortium -v
```

### Test Inference

```bash
curl http://<machine-a-ip>:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "stream": false
  }'
```

---

_Guide version: 1.0_
_Last updated: 2025-12-30_
