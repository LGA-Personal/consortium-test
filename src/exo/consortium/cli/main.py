"""
Consortium CLI - Main entry point

Usage:
    consortium coordinator start [options]
    consortium worker start [options]
    consortium session run [options]
    consortium test [subcommand]
"""

import logging
import signal
import sys

import click

from exo.consortium.coordinator.server import Coordinator
from exo.consortium.worker.server import Worker


def setup_logging(level: str) -> None:
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_layer_range(layers: str) -> tuple[int, int]:
    """Parse layer range string like '0-11' into (start, end)."""
    if not layers:
        return (0, 32)  # Default: all layers
    parts = layers.split("-")
    if len(parts) != 2:
        raise click.BadParameter(f"Invalid layer range: {layers}. Use format 'start-end'")
    return int(parts[0]), int(parts[1])


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Consortium: Distributed Inference with Fuzzy Verification"""
    pass


@cli.group()
def coordinator():
    """Coordinator commands."""
    pass


@coordinator.command("start")
@click.option("--config", type=click.Path(exists=True), help="Config file path")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=50051, type=int, help="Port to listen on")
@click.option("--model", type=click.Path(exists=True), help="Path to GGUF model")
@click.option("--layers", help="Layer range (e.g., '22-32')")
@click.option("--log-level", default="INFO", help="Logging level")
def coordinator_start(config, host, port, model, layers, log_level):
    """Start the coordinator server."""
    setup_logging(log_level)

    click.echo(f"Starting coordinator on {host}:{port}")
    if model:
        click.echo(f"  Model: {model}")
    if layers:
        click.echo(f"  Layers: {layers}")

    # Create and start coordinator
    coord = Coordinator(host=host, port=port)
    coord.start()

    click.echo(click.style("Coordinator running. Press Ctrl+C to stop.", fg="green"))

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        click.echo("\nShutting down coordinator...")
        coord.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Block until terminated
    coord.wait_for_termination()


@cli.group()
def worker():
    """Worker commands."""
    pass


@worker.command("start")
@click.option("--config", type=click.Path(exists=True), help="Config file path")
@click.option("--coordinator", "coordinator_addr", required=True, help="Coordinator address (host:port)")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=50052, type=int, help="Port to listen on")
@click.option("--node-id", help="Node ID (auto-generated if not provided)")
@click.option("--model", type=click.Path(exists=True), help="Path to GGUF model")
@click.option("--layers", required=True, help="Layer range (e.g., '0-11')")
@click.option("--capabilities", default="compute,verify", help="Comma-separated capabilities")
@click.option("--mock", is_flag=True, help="Use mock model (no actual inference)")
@click.option("--log-level", default="INFO", help="Logging level")
def worker_start(config, coordinator_addr, host, port, node_id, model, layers, capabilities, mock, log_level):
    """Start a worker node."""
    setup_logging(log_level)

    layer_start, layer_end = parse_layer_range(layers)
    caps = [c.strip() for c in capabilities.split(",")]

    click.echo(f"Starting worker on {host}:{port}")
    click.echo(f"  Coordinator: {coordinator_addr}")
    click.echo(f"  Model: {model or '(mock mode)'}")
    click.echo(f"  Layers: {layer_start}-{layer_end}")
    click.echo(f"  Capabilities: {caps}")

    # Create worker
    w = Worker(
        node_id=node_id,
        host=host,
        port=port,
        coordinator_address=coordinator_addr,
        capabilities=caps,
    )

    # Load model shard
    use_mock = mock or model is None
    w.load_model_shard(
        model_path=model or "",
        layer_start=layer_start,
        layer_end=layer_end,
        mock_mode=use_mock,
    )

    # Start server
    w.start()

    click.echo(click.style(f"Worker {w.node_id} running. Press Ctrl+C to stop.", fg="green"))

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        click.echo(f"\nShutting down worker {w.node_id}...")
        w.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Block until terminated
    w.wait_for_termination()


@cli.group()
def session():
    """Session commands."""
    pass


@session.command("run")
@click.option("--coordinator", "coordinator_addr", required=True, help="Coordinator address")
@click.option("--prompt", required=True, help="Prompt text")
@click.option("--max-tokens", default=64, type=int, help="Maximum tokens to generate")
@click.option("--seed", default=42, type=int, help="Random seed for sampling")
@click.option("--audit-probability", default=0.2, type=float, help="Audit probability (0-1)")
@click.option("--log-level", default="INFO", help="Logging level")
def session_run(coordinator_addr, prompt, max_tokens, seed, audit_probability, log_level):
    """Run an inference session."""
    setup_logging(log_level)

    click.echo(f"Running session on {coordinator_addr}")
    click.echo(f"  Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    click.echo(f"  Max tokens: {max_tokens}")
    click.echo(f"  Seed: {seed}")
    click.echo(f"  Audit probability: {audit_probability}")

    # For now, use the in-process pipeline for session run
    # Full distributed version would connect to coordinator
    from exo.consortium.core.pipeline import InProcessPipeline
    from exo.consortium.core.session import SessionConfig, create_default_placements

    config = SessionConfig(
        model_id="llama-3-8b",
        rng_seed=seed,
        audit_probability=audit_probability,
        max_tokens=max_tokens,
    )

    placements = create_default_placements()
    prompt_tokens = [1, 2, 3]  # Placeholder tokens

    pipeline = InProcessPipeline.create(
        config=config,
        placements=placements,
        prompt_tokens=prompt_tokens,
        mock_mode=True,
    )

    click.echo("\nGenerating tokens...")

    def on_token(token_idx: int, token_id: int):
        click.echo(f"  Token {token_idx}: {token_id}")

    pipeline.on_token_callback = on_token
    tokens = pipeline.run()
    summary = pipeline.get_summary()

    click.echo(f"\n{click.style('Session complete!', fg='green')}")
    click.echo(f"  Session ID: {summary['session_id']}")
    click.echo(f"  Tokens generated: {summary['tokens_generated']}")
    click.echo(f"  Total time: {summary['duration_ms']:.1f}ms")
    click.echo(f"  Audits: {summary['audits']['passed']} passed, {summary['audits']['failed']} failed")


@session.command("status")
@click.option("--session-id", required=True, help="Session ID to check")
def session_status(session_id):
    """Check session status."""
    click.echo(f"Checking status for session: {session_id}")
    click.echo("  [Requires connection to running coordinator]")


@cli.group()
def test():
    """Testing commands."""
    pass


@test.command("baseline")
@click.option("--model", type=click.Path(exists=True), required=True, help="Path to GGUF model")
@click.option("--prompt", required=True, help="Prompt text")
@click.option("--max-tokens", default=64, type=int, help="Maximum tokens to generate")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--n-gpu-layers", default=-1, type=int, help="GPU layers (-1 = all)")
@click.option("--log-level", default="INFO", help="Logging level")
def test_baseline(model, prompt, max_tokens, seed, n_gpu_layers, log_level):
    """Run single-node baseline using llama-cpp-python directly."""
    setup_logging(log_level)
    import time

    click.echo("Running single-node baseline (full model)...")
    click.echo(f"  Model: {model}")
    click.echo(f"  Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    click.echo(f"  Max tokens: {max_tokens}")
    click.echo(f"  Seed: {seed}")
    click.echo(f"  GPU layers: {n_gpu_layers}")

    try:
        from llama_cpp import Llama
    except ImportError:
        click.echo(click.style("ERROR: llama-cpp-python not installed", fg="red"))
        click.echo("Install with: pip install llama-cpp-python")
        sys.exit(1)

    click.echo("\nLoading model...")
    start_load = time.time()

    llm = Llama(
        model_path=model,
        n_gpu_layers=n_gpu_layers,
        seed=seed,
        n_ctx=2048,
        verbose=False,
    )

    load_time = time.time() - start_load
    click.echo(f"  Model loaded in {load_time:.1f}s")

    click.echo("\nGenerating tokens...")
    start_gen = time.time()

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy for determinism
        top_k=1,
        echo=False,
    )

    gen_time = time.time() - start_gen

    generated_text = output["choices"][0]["text"]
    tokens_generated = output["usage"]["completion_tokens"]
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0

    click.echo(f"\n{click.style('Baseline complete!', fg='green')}")
    click.echo(f"  Tokens generated: {tokens_generated}")
    click.echo(f"  Generation time: {gen_time:.2f}s")
    click.echo(f"  Tokens/sec: {tokens_per_sec:.1f}")
    click.echo(f"\n  Output:\n{generated_text}")


@test.command("inject-fault")
@click.option("--coordinator", "coordinator_addr", required=True, help="Coordinator address")
@click.option("--prompt", required=True, help="Prompt text")
@click.option("--max-tokens", default=64, type=int, help="Maximum tokens")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--kill-node", required=True, help="Node ID to kill")
@click.option("--at-token", default=20, type=int, help="Token index to inject fault")
def test_inject_fault(coordinator_addr, prompt, max_tokens, seed, kill_node, at_token):
    """Run with fault injection for testing failover."""
    click.echo("Running with fault injection...")
    click.echo(f"  Coordinator: {coordinator_addr}")
    click.echo(f"  Kill node: {kill_node} at token {at_token}")
    click.echo("  [Requires running distributed cluster]")


@test.command("e2e")
@click.option("--config", type=click.Path(exists=True), help="Config file for E2E test")
@click.option("--mock", is_flag=True, help="Use mock mode (no model required)")
@click.option("--log-level", default="INFO", help="Logging level")
def test_e2e(config, mock, log_level):
    """Run full end-to-end test suite."""
    setup_logging(log_level)

    click.echo("Running E2E test suite...")

    from exo.consortium.core.pipeline import InProcessPipeline
    from exo.consortium.core.session import SessionConfig, create_default_placements

    # Test configuration
    config_obj = SessionConfig(
        model_id="llama-3-8b",
        rng_seed=42,
        audit_probability=0.2,
        max_tokens=64,
    )

    placements = create_default_placements()
    prompt_tokens = [1, 2, 3]

    click.echo("\n1. Testing in-process pipeline...")
    pipeline = InProcessPipeline.create(
        config=config_obj,
        placements=placements,
        prompt_tokens=prompt_tokens,
        mock_mode=True,
    )

    tokens = pipeline.run()
    summary = pipeline.get_summary()

    click.echo(f"   Generated {summary['tokens_generated']} tokens")
    click.echo(f"   Audits: {summary['audits']['passed']} passed, {summary['audits']['failed']} failed")

    # Verify no fraud
    if summary['audits']['failed'] > 0:
        click.echo(click.style("   FAIL: Audits failed!", fg="red"))
        sys.exit(1)

    click.echo(click.style("   PASS: All audits passed", fg="green"))

    click.echo("\n2. Testing determinism...")
    # Create new pipeline with same config
    pipeline2 = InProcessPipeline.create(
        config=config_obj,
        placements=placements,
        prompt_tokens=prompt_tokens,
        mock_mode=True,
    )
    tokens2 = pipeline2.run()

    if tokens == tokens2:
        click.echo(click.style("   PASS: Deterministic output", fg="green"))
    else:
        click.echo(click.style("   FAIL: Non-deterministic!", fg="red"))
        sys.exit(1)

    click.echo(f"\n{click.style('All E2E tests passed!', fg='green', bold=True)}")


@test.command("quick")
@click.option("--log-level", default="WARNING", help="Logging level")
def test_quick(log_level):
    """Run quick smoke test with mock mode."""
    setup_logging(log_level)

    click.echo("Running quick smoke test...")

    from exo.consortium.core.pipeline import InProcessPipeline
    from exo.consortium.core.session import SessionConfig, create_default_placements

    config = SessionConfig(
        model_id="test",
        rng_seed=42,
        audit_probability=0.2,
        max_tokens=10,
    )

    placements = create_default_placements()
    prompt_tokens = [1, 2, 3]

    pipeline = InProcessPipeline.create(
        config=config,
        placements=placements,
        prompt_tokens=prompt_tokens,
        mock_mode=True,
    )
    tokens = pipeline.run()
    summary = pipeline.get_summary()

    if len(tokens) == 10 and summary['audits']['failed'] == 0:
        click.echo(click.style("PASS", fg="green", bold=True))
    else:
        click.echo(click.style("FAIL", fg="red", bold=True))
        sys.exit(1)


# ========== EXO Integration Commands ==========

@cli.group()
def exo():
    """EXO integration commands for distributed MLX inference."""
    pass


@exo.command("status")
@click.option("--api-url", default="http://localhost:52415", help="EXO API URL")
@click.option("--log-level", default="INFO", help="Logging level")
def exo_status(api_url, log_level):
    """Check EXO cluster status."""
    setup_logging(log_level)
    import json
    import urllib.request
    import urllib.error

    click.echo(f"Checking EXO cluster at {api_url}...")

    try:
        # Check models endpoint
        with urllib.request.urlopen(f"{api_url}/models", timeout=5) as resp:
            models = json.loads(resp.read().decode())

        click.echo(click.style("EXO cluster is running", fg="green"))
        click.echo(f"\nAvailable models:")
        for model in models.get("data", []):
            click.echo(f"  - {model.get('id', 'unknown')}")

    except urllib.error.URLError as e:
        click.echo(click.style(f"Cannot connect to EXO: {e}", fg="red"))
        click.echo("Make sure EXO is running: cd vendor/exo && uv run exo")
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)


@exo.command("chat")
@click.option("--api-url", default="http://localhost:52415", help="EXO API URL")
@click.option("--model", default="mlx-community/Llama-3-8B-Instruct-4bit", help="Model ID")
@click.option("--prompt", required=True, help="Chat prompt")
@click.option("--max-tokens", default=256, type=int, help="Maximum tokens to generate")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature")
@click.option("--coordinator", "coordinator_addr", help="Coordinator address for verification")
@click.option("--session-id", help="Session ID for commitment tracking")
@click.option("--log-level", default="INFO", help="Logging level")
def exo_chat(api_url, model, prompt, max_tokens, temperature, coordinator_addr, session_id, log_level):
    """Send a chat completion request to EXO with verification."""
    setup_logging(log_level)
    import json
    import urllib.request
    import urllib.error

    click.echo(f"Sending chat request to EXO...")
    click.echo(f"  Model: {model}")
    click.echo(f"  Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

    # Build request
    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    try:
        req = urllib.request.Request(
            f"{api_url}/v1/chat/completions",
            data=json.dumps(request_body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())

        # Extract response
        message = result.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        usage = result.get("usage", {})

        click.echo(f"\n{click.style('Response:', fg='green')}")
        click.echo(content)

        click.echo(f"\n{click.style('Usage:', fg='blue')}")
        click.echo(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
        click.echo(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")

        if coordinator_addr:
            click.echo(f"\n{click.style('Verification:', fg='yellow')}")
            click.echo(f"  Coordinator: {coordinator_addr}")
            click.echo(f"  Session ID: {session_id or '(auto-generated)'}")
            click.echo("  Commitments would be forwarded to coordinator")

    except urllib.error.URLError as e:
        click.echo(click.style(f"Request failed: {e}", fg="red"))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)


@exo.command("bridge")
@click.option("--api-url", default="http://localhost:52415", help="EXO API URL")
@click.option("--coordinator", "coordinator_addr", default="localhost:50051", help="Coordinator address")
@click.option("--session-id", required=True, help="Session ID to track")
@click.option("--log-level", default="INFO", help="Logging level")
def exo_bridge(api_url, coordinator_addr, session_id, log_level):
    """Start the coordinator bridge for EXO commitment forwarding."""
    setup_logging(log_level)

    click.echo(f"Starting CoordinatorBridge...")
    click.echo(f"  EXO API: {api_url}")
    click.echo(f"  Coordinator: {coordinator_addr}")
    click.echo(f"  Session ID: {session_id}")

    try:
        from exo.consortium.exo_integration import CoordinatorBridge

        bridge = CoordinatorBridge(
            coordinator_address=coordinator_addr,
            session_id=session_id,
            async_mode=True,
        )
        bridge.start()

        click.echo(click.style("Bridge running. Press Ctrl+C to stop.", fg="green"))

        def signal_handler(sig, frame):
            click.echo("\nStopping bridge...")
            bridge.stop()
            stats = bridge.stats
            click.echo(f"  Sent: {stats['sent']}, Failed: {stats['failed']}")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Keep running
        import time
        while bridge.is_running:
            time.sleep(1)

    except ImportError:
        click.echo(click.style("ERROR: consortium.exo_integration not available", fg="red"))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)


@exo.command("config")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def exo_config(output):
    """Generate example EXO pipeline configuration."""
    import os

    config_template = """# Consortium + EXO Pipeline Configuration
# Generated by: consortium exo config

model:
  id: mlx-community/Llama-3-8B-Instruct-4bit
  n_layers: 32

pipeline:
  stages: 3
  layer_splits: [11, 22]
  backend: ring

verification:
  enabled: true
  interval: 1
  grid_factor: 64
  clamp_min: -100.0
  clamp_max: 100.0

audit:
  probability: 0.2
  timeout_ms: 30000

generation:
  max_tokens: 256
  temperature: 0.7
  top_k: 40
  rng_seed: 42

exo:
  api_url: http://localhost:52415
  timeout_ms: 30000

coordinator:
  address: localhost:50051
  heartbeat_interval_ms: 5000
"""

    if output:
        output_path = os.path.abspath(output)
        with open(output_path, "w") as f:
            f.write(config_template)
        click.echo(f"Configuration written to: {output_path}")
    else:
        click.echo(config_template)


if __name__ == "__main__":
    cli()
