"""
Model Shard Loading and Execution

This module provides the ModelShard class for loading specific layers
of a model and executing forward passes on activation tensors.

For v1, we use llama-cpp-python with the following strategy:
1. Load the full model (required by llama.cpp architecture)
2. Use eval callbacks to extract intermediate activations
3. For partial execution, we inject activations at layer boundaries

Note: This is a PoC implementation. Production would modify llama.cpp
directly for true layer-range execution.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShardConfig:
    """Configuration for a model shard."""

    model_path: str
    layer_start: int  # Inclusive
    layer_end: int  # Exclusive
    hidden_dim: int = 4096  # Llama-3-8B hidden dimension
    n_ctx: int = 2048  # Context size
    n_threads: int = 1  # Single thread for determinism
    n_gpu_layers: int = -1  # All layers on GPU
    seed: int = 42
    verbose: bool = False
    mock_mode: bool = False  # Use mock forward pass for testing


@dataclass
class ForwardResult:
    """Result of a forward pass through the shard."""

    hidden_states: np.ndarray  # Output activation tensor
    compute_time_us: int
    layer_start: int
    layer_end: int

    @property
    def output_activation(self) -> np.ndarray:
        """Alias for hidden_states for backward compatibility."""
        return self.hidden_states


class ModelShard:
    """
    Represents a subset of model layers for pipeline-parallel inference.

    This class wraps llama-cpp-python and provides methods to:
    1. Load the model
    2. Execute forward passes on specific layers
    3. Extract intermediate activations
    4. Manage KV cache for the shard's layers
    """

    def __init__(self, config: ShardConfig):
        """
        Initialize the model shard.

        Args:
            config: Shard configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._activation_cache: Dict[int, np.ndarray] = {}

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    @property
    def num_layers(self) -> int:
        """Number of layers in this shard."""
        return self.config.layer_end - self.config.layer_start

    def load(self) -> None:
        """
        Load the model.

        For v1, we load the full model and will filter layer execution.
        """
        if self._is_loaded:
            logger.warning("Model already loaded")
            return

        # Use mock mode if explicitly requested or no model path
        if self.config.mock_mode or not self.config.model_path:
            self._init_mock_model()
            return

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(
            f"Loading model shard: layers {self.config.layer_start}-{self.config.layer_end}"
        )
        logger.info(f"  Model path: {model_path}")

        try:
            from llama_cpp import Llama

            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                seed=self.config.seed,
                verbose=self.config.verbose,
            )
            self._is_loaded = True
            logger.info("Model loaded successfully")

        except ImportError:
            logger.warning("llama-cpp-python not installed, using mock model")
            self._init_mock_model()

    def _init_mock_model(self) -> None:
        """Initialize a mock model for testing without llama.cpp."""
        logger.info("Initializing mock model for testing")
        self._is_loaded = True
        self._mock_mode = True

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._is_loaded = False
        self._activation_cache.clear()
        logger.info("Model unloaded")

    def encode_tokens(self, token_ids: List[int]) -> np.ndarray:
        """
        Encode tokens to initial embeddings.

        This is used by Stage 0 to convert token IDs to the first
        hidden state before layer execution.

        Args:
            token_ids: List of token IDs

        Returns:
            Embedding tensor of shape [1, seq_len, hidden_dim]
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        if hasattr(self, "_mock_mode") and self._mock_mode:
            # Mock: return random embeddings
            hidden_dim = 4096  # Llama-3-8B
            return np.random.randn(1, len(token_ids), hidden_dim).astype(np.float32)

        # Real implementation would access model.model.embed_tokens
        # For now, we use the model's internal tokenization
        raise NotImplementedError(
            "Token embedding extraction requires llama.cpp modification"
        )

    def forward(
        self,
        input_activation: np.ndarray,
        position_offset: int = 0,
    ) -> ForwardResult:
        """
        Execute forward pass through this shard's layers.

        Args:
            input_activation: Input hidden state [batch, seq_len, hidden_dim]
            position_offset: Position offset for RoPE embeddings

        Returns:
            ForwardResult with output activation and timing
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.perf_counter_ns()

        if hasattr(self, "_mock_mode") and self._mock_mode:
            # Mock: apply a simple transformation
            output = self._mock_forward(input_activation)
        else:
            # Real implementation would:
            # 1. Set up eval callback to capture layer outputs
            # 2. Inject input_activation at layer_start
            # 3. Run layers layer_start to layer_end
            # 4. Extract output from layer_end
            output = self._real_forward(input_activation, position_offset)

        elapsed_ns = time.perf_counter_ns() - start_time
        elapsed_us = elapsed_ns // 1000

        return ForwardResult(
            hidden_states=output,
            compute_time_us=elapsed_us,
            layer_start=self.config.layer_start,
            layer_end=self.config.layer_end,
        )

    def _mock_forward(self, input_activation: np.ndarray) -> np.ndarray:
        """Mock forward pass for testing."""
        # Simple transformation: scale and add noise (deterministic with seed)
        np.random.seed(self.config.seed + self.config.layer_start)

        # Simulate layer computation
        output = input_activation.copy()
        for layer_idx in range(self.config.layer_start, self.config.layer_end):
            # Simple mock: slight modification per layer
            scale = 1.0 + (layer_idx * 0.001)
            output = output * scale

        return output.astype(np.float32)

    def _real_forward(
        self,
        input_activation: np.ndarray,
        position_offset: int,
    ) -> np.ndarray:
        """
        Real forward pass using llama.cpp.

        This requires modifications to llama.cpp to:
        1. Accept pre-computed hidden states as input
        2. Execute only a range of layers
        3. Return the output hidden state

        For v1 PoC, we would need to:
        - Use the eval callback mechanism to intercept tensors
        - Modify the computation graph to skip layers
        """
        raise NotImplementedError(
            "Real forward pass requires llama.cpp modifications. "
            "Use mock mode for testing or implement custom layer execution."
        )

    def compute_logits(self, hidden_state: np.ndarray) -> np.ndarray:
        """
        Compute logits from final hidden state.

        This is used by the final stage to produce token probabilities.

        Args:
            hidden_state: Final hidden state [batch, seq_len, hidden_dim]

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        if hasattr(self, "_mock_mode") and self._mock_mode:
            # Mock: return random logits
            vocab_size = 128256  # Llama-3
            batch, seq_len, _ = hidden_state.shape
            return np.random.randn(batch, seq_len, vocab_size).astype(np.float32)

        raise NotImplementedError(
            "Logit computation requires access to lm_head weights"
        )

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.config.model_path,
            "layer_start": self.config.layer_start,
            "layer_end": self.config.layer_end,
            "num_layers": self.num_layers,
            "is_loaded": self._is_loaded,
            "mock_mode": getattr(self, "_mock_mode", False),
        }


class ModelShardManager:
    """
    Manages multiple model shards for a node that handles multiple stages.

    This is useful for the coordinator which may need to handle
    failover and execute different layer ranges.
    """

    def __init__(self):
        self.shards: Dict[Tuple[int, int], ModelShard] = {}

    def get_or_create_shard(
        self,
        model_path: str,
        layer_start: int,
        layer_end: int,
        **kwargs,
    ) -> ModelShard:
        """
        Get an existing shard or create a new one.

        Args:
            model_path: Path to model file
            layer_start: Start layer (inclusive)
            layer_end: End layer (exclusive)
            **kwargs: Additional ShardConfig parameters

        Returns:
            ModelShard instance
        """
        key = (layer_start, layer_end)

        if key not in self.shards:
            config = ShardConfig(
                model_path=model_path,
                layer_start=layer_start,
                layer_end=layer_end,
                **kwargs,
            )
            shard = ModelShard(config)
            shard.load()
            self.shards[key] = shard

        return self.shards[key]

    def unload_all(self) -> None:
        """Unload all shards."""
        for shard in self.shards.values():
            shard.unload()
        self.shards.clear()

    def get_loaded_ranges(self) -> List[Tuple[int, int]]:
        """Get list of loaded layer ranges."""
        return list(self.shards.keys())
