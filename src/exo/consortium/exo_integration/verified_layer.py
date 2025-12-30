"""
Verified Layer Wrapper for MLX

Wraps MLX model layers to compute canonical-grid commitments
after each forward pass. This enables per-layer verification
in distributed pipeline parallelism.

The wrapper follows EXO's CustomMlxLayer pattern for attribute
delegation, ensuring compatibility with MLX-LM models.
"""

from typing import TYPE_CHECKING, Callable, Protocol

# Type hints for MLX (may not be installed on all systems)
if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn

# Runtime imports are conditional to allow testing without MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    # Create dummy types for non-MLX environments
    class mx:  # type: ignore
        class array:
            pass
    class nn:  # type: ignore
        class Module:
            pass


class _LayerCallable(Protocol):
    """Protocol matching MLX layer callables."""

    def __call__(self, x: "mx.array", *args: object, **kwargs: object) -> "mx.array":
        ...


# Type alias for verification callback
VerificationCallback = Callable[[int, "mx.array"], None]


class VerifiedLayer(nn.Module if MLX_AVAILABLE else object):  # type: ignore
    """
    Wraps an MLX layer to compute verification commitments.

    After each forward pass, invokes a callback with the layer index
    and output hidden state. The callback is responsible for computing
    commitments and accumulating them.

    This follows EXO's CustomMlxLayer pattern for proper attribute
    delegation to the wrapped layer.

    Args:
        original_layer: The MLX layer to wrap
        layer_idx: Global layer index (0 to n_layers-1)
        callback: Function called with (layer_idx, hidden_state) after each forward

    Example:
        ```python
        def record_commitment(layer_idx: int, hidden_state: mx.array):
            commitment = mlx_compute_commitment(hidden_state)
            accumulator.record(layer_idx, commitment)

        verified = VerifiedLayer(layer, layer_idx=5, callback=record_commitment)
        output = verified(x)  # Calls layer, then callback
        ```
    """

    def __init__(
        self,
        original_layer: _LayerCallable,
        layer_idx: int,
        callback: VerificationCallback,
    ):
        if MLX_AVAILABLE:
            super().__init__()

        # Set via object.__setattr__ to avoid __setattr__ recursion
        # This pattern is from EXO's CustomMlxLayer
        object.__setattr__(self, "_original_layer", original_layer)
        object.__setattr__(self, "layer_idx", layer_idx)
        object.__setattr__(self, "callback", callback)

    def __getattr__(self, name: str) -> object:
        """
        Delegate attribute access to the original layer.

        This ensures that model code accessing layer attributes
        (like use_sliding, self_attn, etc.) still works correctly.
        """
        # Avoid recursion for our own attributes
        if name.startswith('_') or name in ('layer_idx', 'callback'):
            return object.__getattribute__(self, name)

        # Try parent class first (for nn.Module attributes)
        if MLX_AVAILABLE:
            try:
                return super().__getattr__(name)
            except AttributeError:
                pass

        # Delegate to original layer
        original_layer = object.__getattribute__(self, "_original_layer")
        return getattr(original_layer, name)

    def __call__(self, x: "mx.array", *args: object, **kwargs: object) -> "mx.array":
        """
        Execute the layer and invoke verification callback.

        Args:
            x: Input hidden state tensor
            *args, **kwargs: Additional arguments passed to the layer

        Returns:
            Output hidden state tensor (unchanged from original layer)
        """
        # Get the original layer
        original_layer: _LayerCallable = object.__getattribute__(self, "_original_layer")

        # Execute the original layer
        output = original_layer(x, *args, **kwargs)

        # Invoke callback with output
        callback: VerificationCallback = object.__getattribute__(self, "callback")
        layer_idx: int = object.__getattribute__(self, "layer_idx")

        # TEMPORARY LOGGING: Show verification is happening
        print(f"🔐 CONSORTIUM: Layer {layer_idx} computing commitment (shape: {output.shape})")

        callback(layer_idx, output)

        return output

    @property
    def original_layer(self) -> _LayerCallable:
        """Access the wrapped original layer."""
        return object.__getattribute__(self, "_original_layer")


def wrap_layers_for_verification(
    layers: list[_LayerCallable],
    start_layer: int,
    callback: VerificationCallback,
) -> list["VerifiedLayer"]:
    """
    Wrap a list of layers with verification.

    Convenience function to wrap all layers in a stage with
    VerifiedLayer wrappers using the same callback.

    Args:
        layers: List of MLX layers to wrap
        start_layer: Global index of the first layer (for correct layer_idx)
        callback: Verification callback to use for all layers

    Returns:
        List of VerifiedLayer wrappers
    """
    return [
        VerifiedLayer(layer, layer_idx=start_layer + i, callback=callback)
        for i, layer in enumerate(layers)
    ]
