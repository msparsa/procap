"""
Integration tests for the ProCap benchmark pipeline.

Tests the full flow from data loading to metric computation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import torch
from pathlib import Path


class TestDataLoading:
    """Tests for data loading utilities."""

    def test_load_csv(self, tmp_path, sample_dataframe):
        """Test loading CSV files."""
        from procap.data.loader import DataLoader

        csv_path = tmp_path / "test.csv"
        sample_dataframe.to_csv(csv_path, index=False)

        df = DataLoader.load(csv_path)

        assert len(df) == len(sample_dataframe)
        assert "sequence" in df.columns

    def test_validate_schema(self, sample_dataframe):
        """Test schema validation."""
        from procap.data.loader import DataLoader

        # Should pass
        assert DataLoader.validate_schema(
            sample_dataframe,
            required_columns=["sequence", "go_terms"],
        )

        # Should fail
        with pytest.raises(ValueError, match="Missing required columns"):
            DataLoader.validate_schema(
                sample_dataframe,
                required_columns=["nonexistent_column"],
            )

    def test_get_info(self, sample_dataframe):
        """Test getting DataFrame info."""
        from procap.data.loader import DataLoader

        info = DataLoader.get_info(sample_dataframe)

        assert info["num_rows"] == 3
        assert "sequence" in info["columns"]
        assert "sequence_stats" in info


class TestTaskRegistry:
    """Tests for task registry."""

    def test_list_tasks(self):
        """Test listing tasks."""
        from procap.tasks.registry import list_tasks

        tasks = list_tasks()

        assert "go_term_identification" in tasks

    def test_get_task(self):
        """Test getting task config."""
        from procap.tasks.registry import get_task

        task = get_task("go_term_identification")

        assert task.name == "go_term_identification"
        # bloom_level might be enum or string depending on pydantic serialization
        bloom_level = task.bloom_level.value if hasattr(task.bloom_level, 'value') else task.bloom_level
        assert bloom_level == "Remembering"
        assert "f1_max" in task.metrics

    def test_filter_by_bloom_level(self):
        """Test filtering by Bloom level."""
        from procap.tasks.registry import TaskRegistry
        from procap.tasks.schemas import BloomLevel

        registry = TaskRegistry()
        remembering_tasks = registry.filter_by_bloom_level(BloomLevel.REMEMBERING)

        assert len(remembering_tasks) > 0
        for task in remembering_tasks:
            assert task.bloom_level == BloomLevel.REMEMBERING


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_f1_max(self):
        """Test f1_max metric."""
        from procap.metrics.classification import f1_max

        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_score = np.array([[0.9, 0.1, 0.8], [0.2, 0.9, 0.1], [0.7, 0.8, 0.2]])

        score = f1_max(y_true, y_score)

        assert 0 <= score <= 1

    def test_auprc_micro(self):
        """Test auprc_micro metric."""
        from procap.metrics.classification import auprc_micro

        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_score = np.array([[0.9, 0.1, 0.8], [0.2, 0.9, 0.1], [0.7, 0.8, 0.2]])

        score = auprc_micro(y_true, y_score)

        assert 0 <= score <= 1

    def test_spearman_correlation(self):
        """Test Spearman correlation."""
        from procap.metrics.regression import spearman_correlation

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

        corr = spearman_correlation(y_true, y_pred)

        assert -1 <= corr <= 1
        assert corr > 0.9  # Should be highly correlated

    def test_rmse(self):
        """Test RMSE metric."""
        from procap.metrics.regression import rmse

        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        error = rmse(y_true, y_pred)

        assert error == 0.0

    def test_rouge_l(self):
        """Test ROUGE-L metric."""
        from procap.metrics.generation import rouge_l

        refs = ["The protein functions in cellular transport"]
        preds = ["The protein functions in cellular transport"]

        score = rouge_l(refs, preds)

        assert score == 1.0  # Perfect match

    def test_novelty_50(self):
        """Test novelty_50 metric."""
        from procap.metrics.generation import novelty_50

        refs = ["Original text here"]
        preds = ["Completely different text that is novel"]

        score = novelty_50(refs, preds)

        assert 0 <= score <= 1


class TestRunners:
    """Tests for task runners."""

    def test_multilabel_runner_extract_targets(self, sample_dataframe):
        """Test GO term extraction in classification runner."""
        from procap.runners.classification import MultiLabelClassificationRunner
        from procap.tasks.schemas import TaskConfig, BloomLevel, TaskType

        task = TaskConfig(
            name="test_task",
            bloom_level=BloomLevel.REMEMBERING,
            domain="Test",
            description="Test task",
            task_type=TaskType.MULTILABEL_CLASSIFICATION,
            dataset_path="test.csv",
            input_fields=["sequence"],
            target_field="go_terms",
            metrics=["f1_max"],
        )

        # Create a mock model
        class MockModel:
            name = "MockModel"
            max_batch_size = 32

            def predict_batch(self, sequences, return_embeddings=False):
                from procap.core.base_model import ModelOutput
                import torch

                return ModelOutput(
                    embeddings=torch.randn(len(sequences), 768),
                    logits=torch.randn(len(sequences), 100),
                )

        runner = MultiLabelClassificationRunner(task, MockModel())
        records = sample_dataframe.to_dict("records")

        targets = runner._extract_targets(records)

        assert len(targets) == 3
        assert isinstance(targets[0], list)
        assert "GO:0005524" in targets[0]


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.slow
    def test_esm_embedding_extraction(self, sample_sequences):
        """Test full ESM embedding extraction pipeline."""
        from procap.models.registry import get_model

        model = get_model("esm2_t6_8M")
        model.load()

        embeddings = model.get_embeddings(sample_sequences)

        assert embeddings.shape[0] == len(sample_sequences)
        assert embeddings.shape[1] == model.hidden_size
        assert not torch.isnan(embeddings).any()

    @pytest.mark.slow
    def test_protbert_embedding_extraction(self, sample_sequences):
        """Test full ProtBERT embedding extraction pipeline."""
        from procap.models.registry import get_model

        model = get_model("protbert")
        model.load()

        embeddings = model.get_embeddings(sample_sequences)

        assert embeddings.shape[0] == len(sample_sequences)
        assert embeddings.shape[1] == model.hidden_size
        assert not torch.isnan(embeddings).any()

    def test_cli_list_models(self):
        """Test CLI list-models command."""
        from typer.testing import CliRunner
        from procap.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["list-models"])

        assert result.exit_code == 0
        assert "esm2_t33_650M" in result.stdout

    def test_cli_list_tasks(self):
        """Test CLI list-tasks command."""
        from typer.testing import CliRunner
        from procap.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["list-tasks"])

        assert result.exit_code == 0
        # Task name may be truncated in table display, so check partial match
        assert "go_term" in result.stdout or "Remembering" in result.stdout
