"""
Configuration Management

This module provides configuration dataclasses and loading functions
for the Consortium system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for the model."""

    model_id: str = "llama3-8b-q4_k_m"
    model_path: Optional[str] = None  # Path to GGUF file
    total_layers: int = 32
    hidden_dim: int = 4096
    vocab_size: int = 128256  # Llama-3 vocab size


@dataclass
class SamplingConfig:
    """Configuration for token sampling."""

    temperature: float = 0.0  # 0.0 for greedy
    top_k: int = 1  # 1 for greedy
    top_p: float = 1.0
    seed: int = 42
    max_tokens: int = 64


@dataclass
class VerificationConfig:
    """Configuration for verification/auditing."""

    audit_probability: float = 0.2  # 20% audit rate
    grid_factor: int = 64
    clamp_min: float = -100.0
    clamp_max: float = 100.0


@dataclass
class NetworkConfig:
    """Configuration for networking."""

    host: str = "0.0.0.0"
    port: int = 50051
    coordinator_address: Optional[str] = None  # For workers: "host:port"
    heartbeat_interval_ms: int = 5000
    stage_timeout_ms: int = 30000
    connect_timeout_ms: int = 10000


@dataclass
class StagePlacement:
    """Placement of a pipeline stage on a node."""

    stage_id: int
    node_id: str
    layer_start: int  # Inclusive
    layer_end: int  # Exclusive

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start


@dataclass
class NodeConfig:
    """Configuration for a node."""

    node_id: str = ""  # Derived from keys if empty
    capabilities: List[str] = field(default_factory=lambda: ["compute", "verify"])
    hardware_desc: str = "unknown"
    layers: Optional[str] = None  # e.g., "0-11" for layers 0-10

    def get_layer_range(self) -> Optional[tuple]:
        """Parse layer range string to (start, end) tuple."""
        if not self.layers:
            return None
        parts = self.layers.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid layer range: {self.layers}")
        return int(parts[0]), int(parts[1])


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "logs"


@dataclass
class ConsortiumConfig:
    """Root configuration for the Consortium system."""

    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    node: NodeConfig = field(default_factory=NodeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: Path) -> "ConsortiumConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: dict) -> "ConsortiumConfig":
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**data.get("model", {})),
            sampling=SamplingConfig(**data.get("sampling", {})),
            verification=VerificationConfig(**data.get("verification", {})),
            network=NetworkConfig(**data.get("network", {})),
            node=NodeConfig(**data.get("node", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_default_placements(num_layers: int = 32) -> List[StagePlacement]:
    """
    Get default 3-stage pipeline placement.

    For Llama-3-8B with 32 layers:
    - Stage 0: layers 0-10 (11 layers)
    - Stage 1: layers 11-21 (11 layers)
    - Stage 2: layers 22-31 (10 layers)
    """
    return [
        StagePlacement(stage_id=0, node_id="node-0", layer_start=0, layer_end=11),
        StagePlacement(stage_id=1, node_id="node-1", layer_start=11, layer_end=22),
        StagePlacement(stage_id=2, node_id="node-2", layer_start=22, layer_end=num_layers),
    ]
