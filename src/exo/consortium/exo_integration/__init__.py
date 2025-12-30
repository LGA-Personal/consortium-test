"""
EXO Integration Module

Provides integration between consortium's verification system
and EXO's MLX-based pipeline parallelism.

This module enables per-layer verification of hidden states during
distributed inference across multiple Apple Silicon devices.
"""

from .verified_layer import VerifiedLayer
from .mlx_canonicalizer import mlx_compute_commitment, mlx_canonicalize
from .commitment_accumulator import CommitmentAccumulator
from .coordinator_bridge import CoordinatorBridge, AsyncCoordinatorBridge

__all__ = [
    "VerifiedLayer",
    "mlx_compute_commitment",
    "mlx_canonicalize",
    "CommitmentAccumulator",
    "CoordinatorBridge",
    "AsyncCoordinatorBridge",
]
