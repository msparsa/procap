"""
Protein-Protein Interaction (PPI) runner for ProCap benchmark.

Supports paired sequence classification for interaction prediction.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from procap.runners.base import BaseRunner
from procap.core.base_model import ModelOutput


class PPIClassificationRunner(BaseRunner):
    """
    Runner for protein-protein interaction prediction.

    Takes two protein sequences as input and predicts whether they interact.

    Embedding combination strategies:
    - Concatenation: [emb_a, emb_b]
    - Hadamard product: emb_a * emb_b
    - Combined: [emb_a, emb_b, emb_a * emb_b]
    """

    def __init__(self, *args, combination: str = "combined", train_ratio: float = 0.8, **kwargs):
        """
        Initialize PPI runner.

        Args:
            combination: How to combine embeddings ("concat", "hadamard", "combined")
            train_ratio: Ratio of data to use for training linear probe
        """
        super().__init__(*args, **kwargs)
        self.combination = combination
        self.train_ratio = train_ratio
        self._head = None  # Optional classification head
        self._linear_probe: Optional[LogisticRegression] = None
        self._use_linear_probe = False

    def run(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Run evaluation with linear probe if no classification head.

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
        """Check if model returns logits or just embeddings."""
        batch_df = df.iloc[:min(2, len(df))]
        batch_records = batch_df.to_dict("records")

        seq_field_a = self.task.input_fields[0]
        seq_field_b = self.task.input_fields[1]

        sequences_a = [r[seq_field_a] for r in batch_records]
        sequences_b = [r[seq_field_b] for r in batch_records]

        output_a: ModelOutput = self.model.predict_batch(sequences_a, return_embeddings=True)
        output_b: ModelOutput = self.model.predict_batch(sequences_b, return_embeddings=True)

        # Check if model returns embeddings without classification head
        self._use_linear_probe = (output_a.logits is None and self._head is None)

        if self._use_linear_probe:
            print("No classification head detected. Using linear probe evaluation.")

    def _train_linear_probe(self, train_df: pd.DataFrame) -> None:
        """Train a linear probe on combined embeddings from training data."""
        print(f"Training linear probe on {len(train_df)} samples...")

        all_embeddings = []
        all_labels = []

        from tqdm import tqdm
        batches = list(self._batch_iterator(train_df))

        for batch_df in tqdm(batches, desc="Extracting train embeddings"):
            batch_records = batch_df.to_dict("records")

            seq_field_a = self.task.input_fields[0]
            seq_field_b = self.task.input_fields[1]

            sequences_a = [r[seq_field_a] for r in batch_records]
            sequences_b = [r[seq_field_b] for r in batch_records]

            # Get embeddings for both proteins
            emb_a = self.model.get_embeddings(sequences_a)
            emb_b = self.model.get_embeddings(sequences_b)

            if isinstance(emb_a, torch.Tensor):
                emb_a = emb_a.cpu().numpy()
            if isinstance(emb_b, torch.Tensor):
                emb_b = emb_b.cpu().numpy()

            # Convert to tensors for combination
            emb_a_tensor = torch.tensor(emb_a) if not isinstance(emb_a, torch.Tensor) else emb_a
            emb_b_tensor = torch.tensor(emb_b) if not isinstance(emb_b, torch.Tensor) else emb_b

            # Combine embeddings
            combined = self._get_combined_embeddings(emb_a_tensor, emb_b_tensor)
            if isinstance(combined, torch.Tensor):
                combined = combined.cpu().numpy()

            all_embeddings.append(combined)

            # Extract labels
            labels = self._extract_targets(batch_records)
            all_labels.extend(labels)

        # Stack all embeddings
        X_train = np.vstack(all_embeddings)
        y_train = np.array(all_labels)

        print(f"Training linear probe: X={X_train.shape}, y={y_train.shape}")

        # Train logistic regression for binary classification
        self._linear_probe = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            n_jobs=-1,
            random_state=42
        )
        self._linear_probe.fit(X_train, y_train)

        # Report training accuracy
        train_acc = self._linear_probe.score(X_train, y_train)
        print(f"Linear probe training accuracy: {train_acc:.4f}")

    def _get_combined_embeddings(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor
    ) -> torch.Tensor:
        """Combine embeddings from two proteins."""
        if self.combination == "concat":
            return torch.cat([emb_a, emb_b], dim=-1)
        elif self.combination == "hadamard":
            return emb_a * emb_b
        elif self.combination == "combined":
            return torch.cat([emb_a, emb_b, emb_a * emb_b], dim=-1)
        else:
            raise ValueError(f"Unknown combination strategy: {self.combination}")

    def _predict_batch(self, batch: List[Dict]) -> List[float]:
        """
        Get interaction probability predictions for batch.

        Returns list of probabilities (float).
        """
        # Extract sequences for both proteins
        seq_field_a = self.task.input_fields[0]  # sequence_a
        seq_field_b = self.task.input_fields[1]  # sequence_b

        sequences_a = [r[seq_field_a] for r in batch]
        sequences_b = [r[seq_field_b] for r in batch]

        # Get embeddings for both sets of sequences
        output_a: ModelOutput = self.model.predict_batch(sequences_a, return_embeddings=True)
        output_b: ModelOutput = self.model.predict_batch(sequences_b, return_embeddings=True)

        if output_a.embeddings is None or output_b.embeddings is None:
            raise ValueError("Model did not return embeddings")

        emb_a = output_a.embeddings
        emb_b = output_b.embeddings

        # Ensure tensors
        if not isinstance(emb_a, torch.Tensor):
            emb_a = torch.tensor(emb_a)
        if not isinstance(emb_b, torch.Tensor):
            emb_b = torch.tensor(emb_b)

        # Combine embeddings
        combined = self._get_combined_embeddings(emb_a, emb_b)

        # If head is attached, use it for prediction
        if self._head is not None:
            logits = self._head(combined)
            if isinstance(logits, torch.Tensor):
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = 1 / (1 + np.exp(-logits))

            # Handle different output shapes
            if probs.ndim == 2:
                if probs.shape[1] == 1:
                    probs = probs[:, 0]
                elif probs.shape[1] == 2:
                    probs = probs[:, 1]  # Probability of positive class

            return probs.tolist()

        # Use linear probe on embeddings
        if self._use_linear_probe and self._linear_probe is not None:
            if isinstance(combined, torch.Tensor):
                combined = combined.cpu().numpy()

            # Get probability predictions from linear probe
            probs = self._linear_probe.predict_proba(combined)
            # Return probability of positive class (column 1)
            if probs.shape[1] == 2:
                return probs[:, 1].tolist()
            else:
                return probs[:, 0].tolist()

        raise ValueError("Model did not return logits and linear probe not trained")

    def _extract_targets(self, batch: List[Dict]) -> List[int]:
        """Extract binary interaction labels."""
        return [int(r[self.task.target_field]) for r in batch]

    def _compute_metrics(
        self,
        predictions: List[float],
        targets: List[int],
        metadata: List[Dict],
    ) -> Dict[str, float]:
        """Compute PPI classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            average_precision_score,
        )

        y_true = np.array(targets)
        y_score = np.array(predictions)
        y_pred = (y_score >= 0.5).astype(int)

        metric_funcs = {
            "accuracy": lambda: accuracy_score(y_true, y_pred),
            "f1": lambda: f1_score(y_true, y_pred, zero_division=0),
            "precision": lambda: precision_score(y_true, y_pred, zero_division=0),
            "recall": lambda: recall_score(y_true, y_pred, zero_division=0),
            "auroc": lambda: roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5,
            "auprc": lambda: average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5,
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


class HomologyDetectionRunner(PPIClassificationRunner):
    """
    Runner for homology detection between protein pairs.

    Inherits from PPIClassificationRunner since both are paired sequence
    binary classification tasks.
    """

    def _compute_metrics(
        self,
        predictions: List[float],
        targets: List[int],
        metadata: List[Dict],
    ) -> Dict[str, float]:
        """Compute homology detection metrics."""
        # Use same metrics as PPI but could add homology-specific metrics
        return super()._compute_metrics(predictions, targets, metadata)
