"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def sample_activation():
    """Generate a sample activation tensor."""
    np.random.seed(42)
    # Typical Llama-3-8B activation shape: [batch, seq_len, hidden_dim]
    return np.random.randn(1, 10, 4096).astype(np.float32)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_activation_with_extremes():
    """Generate activation with extreme values to test clamping."""
    np.random.seed(42)
    arr = np.random.randn(1, 10, 4096).astype(np.float32)
    # Add some extreme values
    arr[0, 0, 0] = 500.0
    arr[0, 0, 1] = -500.0
    arr[0, 0, 2] = np.inf
    arr[0, 0, 3] = -np.inf
    arr[0, 0, 4] = np.nan
    return arr
