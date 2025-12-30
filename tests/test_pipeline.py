"""
Tests for the in-process pipeline execution.
"""

import pytest

from exo.consortium.core.pipeline import InProcessPipeline, PipelineResult
from exo.consortium.core.session import (
    SessionConfig,
    SessionStatus,
    StagePlacement,
    create_default_placements,
)


class TestPipelineCreation:
    """Tests for pipeline initialization."""

    def test_create_pipeline_mock_mode(self):
        """Test creating a pipeline in mock mode."""
        config = SessionConfig(
            max_tokens=10,
            rng_seed=42,
        )
        placements = create_default_placements(num_layers=32, num_stages=3)
        prompt_tokens = [1, 2, 3, 4, 5]

        pipeline = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=prompt_tokens,
            mock_mode=True,
        )

        assert pipeline.session is not None
        assert len(pipeline.executors) == 3
        assert pipeline.session.status == SessionStatus.CREATED

    def test_default_placements(self):
        """Test default placement generation."""
        placements = create_default_placements(num_layers=32, num_stages=3)

        assert len(placements) == 3

        # Check layer distribution: 11, 11, 10
        assert placements[0].layer_start == 0
        assert placements[0].layer_end == 11
        assert placements[1].layer_start == 11
        assert placements[1].layer_end == 22
        assert placements[2].layer_start == 22
        assert placements[2].layer_end == 32

    def test_custom_placements(self):
        """Test with custom placements."""
        placements = [
            StagePlacement(stage_id=0, node_id="worker-0", layer_start=0, layer_end=10),
            StagePlacement(stage_id=1, node_id="worker-1", layer_start=10, layer_end=20),
            StagePlacement(stage_id=2, node_id="coordinator", layer_start=20, layer_end=32),
        ]

        config = SessionConfig(max_tokens=5)
        pipeline = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3],
            mock_mode=True,
        )

        assert len(pipeline.executors) == 3
        assert pipeline.session.placements[0].node_id == "worker-0"
        assert pipeline.session.placements[2].layer_end == 32


class TestPipelineExecution:
    """Tests for pipeline execution."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a pipeline for testing."""
        config = SessionConfig(
            max_tokens=10,
            rng_seed=42,
            audit_probability=0.0,  # Disable audits for basic tests
        )
        placements = create_default_placements(num_layers=32, num_stages=3)

        return InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3, 4, 5],
            mock_mode=True,
        )

    def test_execute_single_token(self, mock_pipeline):
        """Test executing a single token through the pipeline."""
        result = mock_pipeline.execute_token(token_index=0)

        assert isinstance(result, PipelineResult)
        assert result.token_index == 0
        assert result.output_logits is not None
        assert len(result.commitments) == 3  # One per stage
        assert len(result.compute_times_us) == 3
        assert result.total_time_ms > 0

    def test_commitments_are_32_bytes(self, mock_pipeline):
        """Test that commitments are correct size."""
        result = mock_pipeline.execute_token(token_index=0)

        for stage_id, commitment in result.commitments.items():
            assert len(commitment) == 32, f"Stage {stage_id} commitment wrong size"

    def test_receipts_created_per_stage(self, mock_pipeline):
        """Test that receipts are created for each stage."""
        mock_pipeline.execute_token(token_index=0)

        assert len(mock_pipeline.session.receipts) == 3

        for i, receipt in enumerate(mock_pipeline.session.receipts):
            assert receipt.session_id == mock_pipeline.session.id
            assert receipt.token_index == 0
            assert receipt.stage_id == i
            assert len(receipt.commitment) == 32
            assert len(receipt.signature) == 64  # Ed25519 signature


class TestFullInference:
    """Tests for full inference runs."""

    def test_run_full_inference(self):
        """Test running complete inference."""
        config = SessionConfig(
            max_tokens=10,
            rng_seed=42,
            audit_probability=0.0,
        )
        placements = create_default_placements()

        pipeline = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3],
            mock_mode=True,
        )

        tokens = pipeline.run()

        assert len(tokens) == 10
        assert pipeline.session.status == SessionStatus.COMPLETED
        assert len(pipeline.session.receipts) == 30  # 10 tokens * 3 stages

    def test_deterministic_output(self):
        """Test that same seed produces same output."""
        config = SessionConfig(
            max_tokens=5,
            rng_seed=42,
            audit_probability=0.0,
        )
        placements = create_default_placements()

        # Run twice with same seed
        pipeline1 = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3],
            mock_mode=True,
        )
        tokens1 = pipeline1.run()

        pipeline2 = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3],
            mock_mode=True,
        )
        tokens2 = pipeline2.run()

        assert tokens1 == tokens2

    def test_token_callback(self):
        """Test that token callback is invoked."""
        config = SessionConfig(max_tokens=5, audit_probability=0.0)
        placements = create_default_placements()

        pipeline = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3],
            mock_mode=True,
        )

        callback_calls = []
        pipeline.on_token_callback = lambda idx, tok: callback_calls.append((idx, tok))

        pipeline.run()

        assert len(callback_calls) == 5
        assert callback_calls[0][0] == 0
        assert callback_calls[4][0] == 4


class TestAuditing:
    """Tests for audit functionality."""

    def test_audits_occur_with_probability(self):
        """Test that audits occur at expected rate."""
        config = SessionConfig(
            max_tokens=50,
            rng_seed=42,
            audit_probability=0.2,  # 20% audit rate
        )
        placements = create_default_placements()

        pipeline = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3],
            mock_mode=True,
        )

        pipeline.run()

        # Count audited work units
        total_work_units = 50 * 3  # 50 tokens * 3 stages
        audited = sum(
            1
            for tg in pipeline.session.token_generations
            for wu in tg.work_units.values()
            if wu.audited
        )

        # Should be roughly 20% (with some variance)
        assert 15 < audited < 45, f"Expected ~30 audits, got {audited}"

    def test_all_audits_pass_honest_computation(self):
        """Test that all audits pass for honest computation."""
        config = SessionConfig(
            max_tokens=20,
            rng_seed=42,
            audit_probability=0.5,  # Higher rate for better coverage
        )
        placements = create_default_placements()

        pipeline = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3],
            mock_mode=True,
        )

        pipeline.run()

        # All audits should pass
        audit_summary = pipeline.session.get_audit_summary()
        assert audit_summary["failed"] == 0, "No audits should fail for honest computation"
        assert audit_summary["passed"] == audit_summary["total_audited"]

    def test_audit_callback(self):
        """Test that audit callback is invoked."""
        config = SessionConfig(
            max_tokens=10,
            rng_seed=42,
            audit_probability=1.0,  # Audit everything
        )
        placements = create_default_placements()

        pipeline = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3],
            mock_mode=True,
        )

        audit_results = []
        pipeline.on_audit_callback = (
            lambda token_idx, stage_id, passed: audit_results.append(
                {"token": token_idx, "stage": stage_id, "passed": passed}
            )
        )

        pipeline.run()

        # Should have 30 audit results (10 tokens * 3 stages)
        assert len(audit_results) == 30
        assert all(r["passed"] for r in audit_results)


class TestSessionSummary:
    """Tests for session summary and metrics."""

    def test_get_summary(self):
        """Test getting pipeline summary."""
        config = SessionConfig(max_tokens=5, audit_probability=0.2)
        placements = create_default_placements()

        pipeline = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3],
            mock_mode=True,
        )

        pipeline.run()
        summary = pipeline.get_summary()

        assert summary["session_id"] == pipeline.session.id
        assert summary["status"] == "completed"
        assert summary["tokens_generated"] == 5
        assert summary["duration_ms"] > 0
        assert summary["receipts"] == 15
        assert len(summary["stages"]) == 3

    def test_tokens_per_second(self):
        """Test tokens per second calculation."""
        config = SessionConfig(max_tokens=10)
        placements = create_default_placements()

        pipeline = InProcessPipeline.create(
            config=config,
            placements=placements,
            prompt_tokens=[1, 2, 3],
            mock_mode=True,
        )

        pipeline.run()

        tps = pipeline.session.get_tokens_per_second()
        assert tps is not None
        assert tps > 0
