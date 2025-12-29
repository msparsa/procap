"""
Token classification runner for ProCap benchmark.

Supports per-residue classification tasks like secondary structure prediction.
Uses linear probe evaluation when no classification head is attached.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from procap.runners.base import BaseRunner
from procap.core.base_model import ModelOutput


class TokenClassificationRunner(BaseRunner):
    """
    Runner for token (per-residue) classification tasks.

    Used for tasks like secondary structure prediction where each residue
    gets a class label (e.g., H/E/C for SS3).

    Uses linear probe evaluation: trains a simple logistic regression
    on embeddings when no classification head is attached.
    """

    # Class mappings for different tasks
    SS3_CLASSES = {"H": 0, "E": 1, "C": 2}
    SS8_CLASSES = {"H": 0, "G": 1, "I": 2, "E": 3, "B": 4, "T": 5, "S": 6, "C": 7}

    def __init__(self, *args, num_classes: int = 3, train_ratio: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self._label_map = self.SS3_CLASSES if num_classes == 3 else self.SS8_CLASSES
        self.train_ratio = train_ratio
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
        sequences = [r[self.task.input_fields[0]] for r in batch_records]

        output: ModelOutput = self.model.predict_batch(sequences, return_embeddings=True)
        self._use_linear_probe = (output.logits is None)

        if self._use_linear_probe:
            print("No classification head detected. Using linear probe evaluation.")

    def _train_linear_probe(self, train_df: pd.DataFrame) -> None:
        """Train a linear probe on training embeddings."""
        print(f"Training linear probe on {len(train_df)} samples...")

        all_embeddings = []
        all_labels = []

        from tqdm import tqdm
        batches = list(self._batch_iterator(train_df))

        for batch_df in tqdm(batches, desc="Extracting train embeddings"):
            batch_records = batch_df.to_dict("records")
            sequences = [r[self.task.input_fields[0]] for r in batch_records]

            # Get per-residue embeddings
            embeddings = self.model.get_embeddings(sequences, pooling="per_residue")
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Extract labels
            targets = self._extract_targets(batch_records)

            # Flatten embeddings and labels for training
            for i, (emb, labels) in enumerate(zip(embeddings, targets)):
                seq_len = len(labels)
                # Handle sequence length mismatch (due to special tokens)
                if emb.shape[0] > seq_len:
                    emb = emb[1:seq_len+1]  # Skip CLS token
                elif emb.shape[0] < seq_len:
                    emb = emb[:seq_len]

                # Ensure shapes match
                min_len = min(emb.shape[0], seq_len)
                all_embeddings.append(emb[:min_len])
                all_labels.extend(labels[:min_len])

        # Stack all embeddings
        X_train = np.vstack(all_embeddings)
        y_train = np.array(all_labels)

        print(f"Training linear probe: X={X_train.shape}, y={y_train.shape}")

        # Train logistic regression
        self._linear_probe = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            n_jobs=-1,
            random_state=42
        )
        self._linear_probe.fit(X_train, y_train)

        # Report training accuracy
        train_acc = self._linear_probe.score(X_train, y_train)
        print(f"Linear probe training accuracy: {train_acc:.4f}")

    def _predict_batch(self, batch: List[Dict]) -> List[np.ndarray]:
        """
        Get per-residue predictions for batch.

        Returns list of arrays, each with shape [seq_len, num_classes].
        """
        sequences = [r[self.task.input_fields[0]] for r in batch]

        # Get model output
        output: ModelOutput = self.model.predict_batch(sequences, return_embeddings=True)

        if output.logits is not None and not self._use_linear_probe:
            # Model has classification head attached
            if isinstance(output.logits, torch.Tensor):
                logits = output.logits.cpu().numpy()
            else:
                logits = output.logits
            return [logits[i] for i in range(len(logits))]

        # Use linear probe on embeddings
        if self._use_linear_probe and self._linear_probe is not None:
            # Get per-residue embeddings
            embeddings = self.model.get_embeddings(sequences, pooling="per_residue")
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            predictions = []
            for i, seq in enumerate(sequences):
                seq_len = len(seq)
                emb = embeddings[i]

                # Handle sequence length mismatch
                if emb.shape[0] > seq_len:
                    emb = emb[1:seq_len+1]  # Skip CLS token

                min_len = min(emb.shape[0], seq_len)
                emb = emb[:min_len]

                # Get predictions from linear probe
                probs = self._linear_probe.predict_proba(emb)
                predictions.append(probs)

            return predictions

        raise ValueError("Model did not return logits and linear probe not trained")

    def _extract_targets(self, batch: List[Dict]) -> List[List[int]]:
        """
        Extract per-residue labels from label strings.

        "HHHEEECCC" -> [0, 0, 0, 1, 1, 1, 2, 2, 2] for SS3
        """
        targets = []
        for r in batch:
            label_str = r[self.task.target_field]
            if isinstance(label_str, str):
                labels = [self._label_map.get(c, 2) for c in label_str]  # Default to C/coil
            else:
                labels = list(label_str)
            targets.append(labels)
        return targets

    def _compute_metrics(
        self,
        predictions: List[np.ndarray],
        targets: List[List[int]],
        metadata: List[Dict],
    ) -> Dict[str, float]:
        """Compute per-residue classification metrics."""
        from sklearn.metrics import accuracy_score, f1_score

        # Flatten predictions and targets
        all_preds = []
        all_targets = []

        for pred_logits, target_labels in zip(predictions, targets):
            # pred_logits: [seq_len, num_classes]
            # target_labels: [seq_len]
            seq_len = len(target_labels)

            # Truncate predictions to match target length
            if len(pred_logits) > seq_len:
                pred_logits = pred_logits[:seq_len]
            elif len(pred_logits) < seq_len:
                # Pad with zeros if needed
                padding = np.zeros((seq_len - len(pred_logits), self.num_classes))
                pred_logits = np.vstack([pred_logits, padding])

            # Get predicted classes
            pred_classes = np.argmax(pred_logits, axis=-1)

            all_preds.extend(pred_classes.tolist())
            all_targets.extend(target_labels)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        results = {}

        if "accuracy" in self.task.metrics:
            results["accuracy"] = accuracy_score(all_targets, all_preds)

        if "f1_macro" in self.task.metrics:
            results["f1_macro"] = f1_score(
                all_targets, all_preds, average="macro", zero_division=0
            )

        if "f1_micro" in self.task.metrics:
            results["f1_micro"] = f1_score(
                all_targets, all_preds, average="micro", zero_division=0
            )

        # Per-class accuracy
        for class_name, class_idx in self._label_map.items():
            mask = all_targets == class_idx
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == class_idx).mean()
                results[f"accuracy_{class_name}"] = float(class_acc)

        return results
