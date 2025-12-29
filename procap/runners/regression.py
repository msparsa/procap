"""
Regression runner for ProCap benchmark.

Supports regression tasks like:
- Variant effect prediction
- Binding affinity prediction
- Stability change prediction (DDG)
"""

from typing import Any, Dict, List, Optional
import pandas as pd

import numpy as np
from sklearn.linear_model import Ridge

from procap.runners.base import BaseRunner
from procap.core.base_model import ModelOutput


class RegressionRunner(BaseRunner):
    """
    Runner for regression tasks.

    Used for tasks predicting continuous values like effect scores,
    binding affinities, or stability changes.
    """

    def __init__(self, *args, train_ratio: float = 0.8, **kwargs):
        """
        Initialize regression runner.

        Args:
            train_ratio: Ratio of data to use for training linear probe
        """
        super().__init__(*args, **kwargs)
        self.train_ratio = train_ratio
        self._linear_probe: Optional[Ridge] = None
        self._use_linear_probe = False

    def run(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Run evaluation with linear probe if no regression head.

        Splits data into train/test, trains linear probe on train embeddings,
        evaluates on test set.
        """
        # Validate input data
        self._validate_data(df)

        # Split data for linear probe training
        n_samples = len(df)
        n_train = int(n_samples * self.train_ratio)

        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        # First, check if we need linear probe by testing one batch
        self._check_if_linear_probe_needed(train_df)

        if self._use_linear_probe:
            # Train linear probe on training data
            self._train_linear_probe(train_df)

        # Evaluate on test data
        all_predictions = []
        all_targets = []
        all_metadata = []

        from tqdm import tqdm
        batches = list(self._batch_iterator(test_df))
        iterator = (
            tqdm(batches, desc=f"Evaluating {self.task.name}")
            if self.show_progress
            else batches
        )

        for batch_df in iterator:
            batch_records = batch_df.to_dict("records")
            predictions = self._predict_batch(batch_records)
            all_predictions.extend(predictions)
            targets = self._extract_targets(batch_records)
            all_targets.extend(targets)
            metadata = self._extract_metadata(batch_records)
            all_metadata.extend(metadata)

        return self._compute_metrics(all_predictions, all_targets, all_metadata)

    def _check_if_linear_probe_needed(self, df: pd.DataFrame) -> None:
        """Check if model returns predictions/logits or just embeddings."""
        batch_df = df.iloc[:min(2, len(df))]
        batch_records = batch_df.to_dict("records")
        sequences = [r[self.task.input_fields[0]] for r in batch_records]

        output: ModelOutput = self.model.predict_batch(sequences, return_embeddings=True)

        # Check if model returns predictions or logits
        self._use_linear_probe = (output.predictions is None and output.logits is None)

        if self._use_linear_probe:
            print("No regression head detected. Using linear probe evaluation.")

    def _train_linear_probe(self, train_df: pd.DataFrame) -> None:
        """Train a linear probe (Ridge regression) on training embeddings."""
        print(f"Training linear probe on {len(train_df)} samples...")

        all_embeddings = []
        all_labels = []

        from tqdm import tqdm
        batches = list(self._batch_iterator(train_df))

        for batch_df in tqdm(batches, desc="Extracting train embeddings"):
            batch_records = batch_df.to_dict("records")
            sequences = [r[self.task.input_fields[0]] for r in batch_records]

            # Get embeddings
            embeddings = self.model.get_embeddings(sequences)

            import torch
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            all_embeddings.append(embeddings)

            # Extract labels
            labels = self._extract_targets(batch_records)
            all_labels.extend(labels)

        # Stack all embeddings
        X_train = np.vstack(all_embeddings)
        y_train = np.array(all_labels)

        print(f"Training linear probe: X={X_train.shape}, y={y_train.shape}")

        # Train Ridge regression
        self._linear_probe = Ridge(alpha=1.0, random_state=42)
        self._linear_probe.fit(X_train, y_train)

        # Report training R2 score
        train_r2 = self._linear_probe.score(X_train, y_train)
        print(f"Linear probe training R2: {train_r2:.4f}")

    def _predict_batch(self, batch: List[Dict]) -> List[float]:
        """Get regression predictions for batch."""
        sequences = [r[self.task.input_fields[0]] for r in batch]
        output: ModelOutput = self.model.predict_batch(sequences, return_embeddings=True)

        # Use predictions if available (from regression head)
        if output.predictions is not None and not self._use_linear_probe:
            import torch

            if isinstance(output.predictions, torch.Tensor):
                preds = output.predictions.cpu().numpy()
            else:
                preds = output.predictions

            # Flatten if needed
            if preds.ndim == 2:
                preds = preds[:, 0]
            return preds.tolist()

        # Fall back to logits
        if output.logits is not None and not self._use_linear_probe:
            import torch

            if isinstance(output.logits, torch.Tensor):
                preds = output.logits.cpu().numpy()
            else:
                preds = output.logits

            if preds.ndim == 2:
                preds = preds[:, 0]
            return preds.tolist()

        # Use linear probe on embeddings
        if self._use_linear_probe and self._linear_probe is not None:
            # Get embeddings
            embeddings = self.model.get_embeddings(sequences)

            import torch
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Get predictions from linear probe
            preds = self._linear_probe.predict(embeddings)
            return preds.tolist()

        raise ValueError("Model did not return predictions/logits and linear probe not trained")

    def _extract_targets(self, batch: List[Dict]) -> List[float]:
        """Extract target values."""
        targets = []
        for r in batch:
            value = r[self.task.target_field]
            if isinstance(value, (int, float)):
                targets.append(float(value))
            else:
                # Try to parse
                try:
                    targets.append(float(value))
                except (ValueError, TypeError):
                    targets.append(0.0)
        return targets

    def _compute_metrics(
        self,
        predictions: List[float],
        targets: List[float],
        metadata: List[Dict],
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        from procap.metrics.regression import (
            spearman_correlation,
            pearson_correlation,
            rmse,
            mae,
            r2_score,
        )

        y_true = np.array(targets)
        y_pred = np.array(predictions)

        metric_funcs = {
            "spearman": lambda: spearman_correlation(y_true, y_pred),
            "pearson": lambda: pearson_correlation(y_true, y_pred),
            "rmse": lambda: rmse(y_true, y_pred),
            "mae": lambda: mae(y_true, y_pred),
            "r2": lambda: r2_score(y_true, y_pred),
        }

        results = {}
        for metric_name in self.task.metrics:
            if metric_name in metric_funcs:
                try:
                    results[metric_name] = metric_funcs[metric_name]()
                except Exception as e:
                    print(f"Warning: Could not compute {metric_name}: {e}")
                    results[metric_name] = 0.0

        return results
