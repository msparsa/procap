"""
Base runner class for task evaluation.

Provides common functionality for all task runners:
- Batch iteration
- Progress tracking
- Metric computation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from tqdm import tqdm

from procap.core.base_model import BaseProteinModel
from procap.tasks.schemas import TaskConfig


class BaseRunner(ABC):
    """
    Abstract base runner with efficient batching support.

    All task-specific runners should inherit from this class.

    Usage:
        runner = MultiLabelClassificationRunner(task_config, model)
        results = runner.run(dataframe)
    """

    def __init__(
        self,
        task: TaskConfig,
        model: BaseProteinModel,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ):
        """
        Initialize runner.

        Args:
            task: Task configuration
            model: Loaded protein language model
            batch_size: Batch size (overrides task default)
            show_progress: Whether to show progress bar
        """
        self.task = task
        self.model = model
        self.batch_size = batch_size or task.batch_size
        self.show_progress = show_progress

    def run(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Run the task evaluation.

        Args:
            df: DataFrame with input data

        Returns:
            Dictionary mapping metric names to scores
        """
        # Validate input data
        self._validate_data(df)

        # Run predictions with batching
        all_predictions = []
        all_targets = []
        all_metadata = []

        batches = list(self._batch_iterator(df))
        iterator = (
            tqdm(batches, desc=f"Evaluating {self.task.name}")
            if self.show_progress
            else batches
        )

        for batch_df in iterator:
            batch_records = batch_df.to_dict("records")

            # Get model predictions
            predictions = self._predict_batch(batch_records)
            all_predictions.extend(predictions)

            # Extract targets
            targets = self._extract_targets(batch_records)
            all_targets.extend(targets)

            # Extract metadata for fairness metrics
            metadata = self._extract_metadata(batch_records)
            all_metadata.extend(metadata)

        # Compute metrics
        return self._compute_metrics(all_predictions, all_targets, all_metadata)

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist.

        Raises:
            ValueError: If required columns are missing
        """
        required = self.task.input_fields + [self.task.target_field]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _batch_iterator(
        self, df: pd.DataFrame
    ) -> Generator[pd.DataFrame, None, None]:
        """Yield batches of data."""
        for start in range(0, len(df), self.batch_size):
            yield df.iloc[start : start + self.batch_size]

    @abstractmethod
    def _predict_batch(self, batch: List[Dict]) -> List[Any]:
        """
        Get predictions for a batch of records.

        Args:
            batch: List of record dictionaries

        Returns:
            List of predictions
        """
        pass

    @abstractmethod
    def _extract_targets(self, batch: List[Dict]) -> List[Any]:
        """
        Extract ground truth targets from batch.

        Args:
            batch: List of record dictionaries

        Returns:
            List of target values
        """
        pass

    def _extract_metadata(self, batch: List[Dict]) -> List[Dict]:
        """
        Extract metadata for fairness metrics.

        Args:
            batch: List of record dictionaries

        Returns:
            List of metadata dictionaries
        """
        return [
            {
                "family": r.get("family"),
                "species": r.get("species"),
            }
            for r in batch
        ]

    @abstractmethod
    def _compute_metrics(
        self,
        predictions: List[Any],
        targets: List[Any],
        metadata: List[Dict],
    ) -> Dict[str, float]:
        """
        Compute all specified metrics.

        Args:
            predictions: List of model predictions
            targets: List of ground truth values
            metadata: List of metadata dictionaries

        Returns:
            Dictionary mapping metric names to scores
        """
        pass
