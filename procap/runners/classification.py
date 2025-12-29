"""
Classification runners for ProCap benchmark.

Supports:
- Multi-label classification (GO term prediction)
- Binary classification (PPI prediction)
- Multiclass classification (protein family classification)

Uses linear probe evaluation when no classification head is attached.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from procap.runners.base import BaseRunner
from procap.core.base_model import ModelOutput


class MultiLabelClassificationRunner(BaseRunner):
    """
    Runner for multi-label classification tasks.

    Used for GO term prediction where each protein can have multiple GO terms.

    Handles:
    - Parsing semicolon-separated GO terms
    - Converting predictions to label space
    - Computing multi-label metrics
    - Linear probe evaluation when no classification head is attached
    """

    def __init__(self, *args, train_ratio: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self._label_binarizer: MultiLabelBinarizer = None
        self._all_labels: List[str] = []
        self.train_ratio = train_ratio
        self._linear_probe: Optional[OneVsRestClassifier] = None
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

            # Get sequence-level embeddings
            embeddings = self.model.get_embeddings(sequences, pooling="mean")
            import torch
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Extract labels
            targets = self._extract_targets(batch_records)

            all_embeddings.append(embeddings)
            all_labels.extend(targets)

        # Stack all embeddings
        X_train = np.vstack(all_embeddings)

        # Fit label binarizer on all observed labels during training
        unique_labels = sorted(set([label for labels in all_labels for label in labels]))
        self._label_binarizer = MultiLabelBinarizer(classes=unique_labels)
        y_train = self._label_binarizer.fit_transform(all_labels)

        print(f"Training linear probe: X={X_train.shape}, y={y_train.shape}")

        # Train one-vs-rest logistic regression for multi-label
        self._linear_probe = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                n_jobs=-1,
                random_state=42
            )
        )
        self._linear_probe.fit(X_train, y_train)

        # Report training accuracy
        train_preds = self._linear_probe.predict(X_train)
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_train, train_preds)
        print(f"Linear probe training accuracy: {train_acc:.4f}")

    def _predict_batch(self, batch: List[Dict]) -> List[np.ndarray]:
        """
        Get probability predictions for batch.

        Returns list of probability arrays.
        """
        # Extract sequences from input fields
        sequences = [r[self.task.input_fields[0]] for r in batch]

        # Get model output
        output: ModelOutput = self.model.predict_batch(sequences, return_embeddings=True)

        # If model has logits (classification head attached)
        if output.logits is not None and not self._use_linear_probe:
            # Convert logits to probabilities using sigmoid (multi-label)
            import torch

            if isinstance(output.logits, torch.Tensor):
                probs = torch.sigmoid(output.logits).cpu().numpy()
            else:
                probs = 1 / (1 + np.exp(-output.logits))  # numpy sigmoid
            return [probs[i] for i in range(len(probs))]

        # Use linear probe on embeddings
        if self._use_linear_probe and self._linear_probe is not None:
            # Get sequence-level embeddings
            embeddings = self.model.get_embeddings(sequences, pooling="mean")
            import torch
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Get predictions from linear probe (probabilities for each label)
            probs = self._linear_probe.predict_proba(embeddings)

            # OneVsRestClassifier returns list of arrays, need to combine them
            # Each estimator gives probabilities for binary decision
            # We want the probability of the positive class for each label
            if isinstance(probs, list):
                # Stack probabilities for positive class from each binary classifier
                probs = np.column_stack([p[:, 1] if p.shape[1] == 2 else p for p in probs])

            return [probs[i] for i in range(len(probs))]

        raise ValueError("Model did not return logits and linear probe not trained")

    def _extract_targets(self, batch: List[Dict]) -> List[List[str]]:
        """
        Extract GO terms from semicolon-separated string.

        "GO:0001;GO:0002" -> ["GO:0001", "GO:0002"]
        """
        targets = []
        for r in batch:
            target_str = r[self.task.target_field]
            if isinstance(target_str, str):
                labels = [t.strip() for t in target_str.split(";") if t.strip()]
            elif isinstance(target_str, list):
                labels = target_str
            else:
                labels = []
            targets.append(labels)

            # Collect all labels for binarizer (only during training if using linear probe)
            if not self._use_linear_probe:
                self._all_labels.extend(labels)

        return targets

    def _compute_metrics(
        self,
        predictions: List[np.ndarray],
        targets: List[List[str]],
        metadata: List[Dict],
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        from procap.metrics.classification import (
            f1_max,
            auprc_micro,
            auroc_micro,
            f1_micro,
            precision_micro,
            recall_micro,
        )

        # If using linear probe, label binarizer is already fitted
        if not self._use_linear_probe:
            # Fit label binarizer on all observed labels
            unique_labels = sorted(set(self._all_labels))
            self._label_binarizer = MultiLabelBinarizer(classes=unique_labels)
            y_true = self._label_binarizer.fit_transform(targets)
        else:
            # Use already fitted binarizer from training
            y_true = self._label_binarizer.transform(targets)

        # Stack predictions
        y_score = np.vstack(predictions)

        # Handle dimension mismatch between predictions and labels
        n_labels = y_true.shape[1]
        n_preds = y_score.shape[1]

        if n_preds != n_labels:
            if n_preds < n_labels:
                # Pad predictions with zeros
                padding = np.zeros((y_score.shape[0], n_labels - n_preds))
                y_score = np.hstack([y_score, padding])
            else:
                # Truncate predictions
                y_score = y_score[:, :n_labels]

        # Compute requested metrics
        metric_funcs = {
            "f1_max": f1_max,
            "auprc_micro": auprc_micro,
            "auroc_micro": auroc_micro,
            "f1_micro": f1_micro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
        }

        results = {}
        for metric_name in self.task.metrics:
            if metric_name in metric_funcs:
                try:
                    results[metric_name] = metric_funcs[metric_name](y_true, y_score)
                except Exception as e:
                    print(f"Warning: Could not compute {metric_name}: {e}")
                    results[metric_name] = 0.0

        return results


class MulticlassClassificationRunner(BaseRunner):
    """
    Runner for multiclass classification tasks.

    Used for tasks like protein family classification where each protein
    belongs to exactly one class.

    Uses linear probe evaluation when no classification head is attached.
    """

    def __init__(self, *args, train_ratio: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self._label_encoder = None
        self._classes: List[str] = []
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

            # Get sequence-level embeddings
            embeddings = self.model.get_embeddings(sequences, pooling="mean")
            import torch
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Extract labels
            targets = self._extract_targets(batch_records)

            all_embeddings.append(embeddings)
            all_labels.extend(targets)

        # Stack all embeddings
        X_train = np.vstack(all_embeddings)

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self._label_encoder = LabelEncoder()
        y_train = self._label_encoder.fit_transform(all_labels)

        print(f"Training linear probe: X={X_train.shape}, y={y_train.shape}")
        print(f"Number of classes: {len(self._label_encoder.classes_)}")

        # Train multinomial logistic regression for multiclass
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
        """Get probability predictions for batch."""
        sequences = [r[self.task.input_fields[0]] for r in batch]
        output: ModelOutput = self.model.predict_batch(sequences, return_embeddings=True)

        if output.logits is not None and not self._use_linear_probe:
            import torch

            if isinstance(output.logits, torch.Tensor):
                # Softmax for multiclass
                probs = torch.softmax(output.logits, dim=-1).cpu().numpy()
            else:
                # numpy softmax
                exp_logits = np.exp(output.logits - np.max(output.logits, axis=-1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return [probs[i] for i in range(len(probs))]

        # Use linear probe on embeddings
        if self._use_linear_probe and self._linear_probe is not None:
            # Get sequence-level embeddings
            embeddings = self.model.get_embeddings(sequences, pooling="mean")
            import torch
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Get predictions from linear probe
            probs = self._linear_probe.predict_proba(embeddings)
            return [probs[i] for i in range(len(probs))]

        raise ValueError("Model did not return logits and linear probe not trained")

    def _extract_targets(self, batch: List[Dict]) -> List[str]:
        """Extract class labels."""
        targets = []
        for r in batch:
            label = r[self.task.target_field]
            targets.append(str(label))
            if not self._use_linear_probe and label not in self._classes:
                self._classes.append(label)
        return targets

    def _compute_metrics(
        self,
        predictions: List[np.ndarray],
        targets: List[str],
        metadata: List[Dict],
    ) -> Dict[str, float]:
        """Compute multiclass classification metrics."""
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, f1_score

        # If using linear probe, label encoder is already fitted
        if not self._use_linear_probe:
            # Encode labels
            self._label_encoder = LabelEncoder()
            self._label_encoder.fit(self._classes)

        # Handle unseen labels in test set by filtering them out
        # This can happen when we have many classes and small sample sizes
        valid_indices = []
        filtered_targets = []
        for i, target in enumerate(targets):
            if target in self._label_encoder.classes_:
                valid_indices.append(i)
                filtered_targets.append(target)

        if len(valid_indices) == 0:
            print("Warning: No valid test samples with known classes")
            return {metric: 0.0 for metric in self.task.metrics}

        if len(valid_indices) < len(targets):
            print(f"Warning: {len(targets) - len(valid_indices)} test samples with unseen classes were excluded")

        y_true = self._label_encoder.transform(filtered_targets)

        # Filter predictions to only include valid samples
        predictions_filtered = [predictions[i] for i in valid_indices]

        # Get predicted classes (using filtered predictions)
        y_score = np.vstack(predictions_filtered)
        n_classes = len(self._label_encoder.classes_)

        # Handle dimension mismatch
        if y_score.shape[1] != n_classes:
            if y_score.shape[1] < n_classes:
                padding = np.zeros((y_score.shape[0], n_classes - y_score.shape[1]))
                y_score = np.hstack([y_score, padding])
            else:
                y_score = y_score[:, :n_classes]

        y_pred = np.argmax(y_score, axis=1)

        results = {}
        if "accuracy" in self.task.metrics:
            results["accuracy"] = accuracy_score(y_true, y_pred)
        if "f1_macro" in self.task.metrics:
            results["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if "f1_micro" in self.task.metrics:
            results["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

        return results


class BinaryClassificationRunner(BaseRunner):
    """
    Runner for binary classification tasks.

    Used for tasks like protein-protein interaction prediction.

    Uses linear probe evaluation when no classification head is attached.
    """

    def __init__(self, *args, train_ratio: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
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

            # Get sequence-level embeddings
            embeddings = self.model.get_embeddings(sequences, pooling="mean")
            import torch
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Extract labels
            targets = self._extract_targets(batch_records)

            all_embeddings.append(embeddings)
            all_labels.extend(targets)

        # Stack all embeddings
        X_train = np.vstack(all_embeddings)
        y_train = np.array(all_labels)

        print(f"Training linear probe: X={X_train.shape}, y={y_train.shape}")

        # Train binary logistic regression
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

    def _predict_batch(self, batch: List[Dict]) -> List[float]:
        """Get probability predictions for batch."""
        sequences = [r[self.task.input_fields[0]] for r in batch]
        output: ModelOutput = self.model.predict_batch(sequences, return_embeddings=True)

        if output.logits is not None and not self._use_linear_probe:
            import torch

            if isinstance(output.logits, torch.Tensor):
                probs = torch.sigmoid(output.logits).cpu().numpy()
            else:
                probs = 1 / (1 + np.exp(-output.logits))

            # Handle different output shapes
            if probs.ndim == 1:
                return probs.tolist()
            elif probs.ndim == 2 and probs.shape[1] == 1:
                return probs[:, 0].tolist()
            elif probs.ndim == 2 and probs.shape[1] == 2:
                return probs[:, 1].tolist()  # Probability of positive class
            else:
                return probs[:, 0].tolist()

        # Use linear probe on embeddings
        if self._use_linear_probe and self._linear_probe is not None:
            # Get sequence-level embeddings
            embeddings = self.model.get_embeddings(sequences, pooling="mean")
            import torch
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Get predictions from linear probe
            probs = self._linear_probe.predict_proba(embeddings)
            # Return probability of positive class
            return probs[:, 1].tolist()

        raise ValueError("Model did not return logits and linear probe not trained")

    def _extract_targets(self, batch: List[Dict]) -> List[int]:
        """Extract binary labels."""
        return [int(r[self.task.target_field]) for r in batch]

    def _compute_metrics(
        self,
        predictions: List[float],
        targets: List[int],
        metadata: List[Dict],
    ) -> Dict[str, float]:
        """Compute binary classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_true = np.array(targets)
        y_score = np.array(predictions)
        y_pred = (y_score >= 0.5).astype(int)

        metric_funcs = {
            "accuracy": lambda: accuracy_score(y_true, y_pred),
            "f1": lambda: f1_score(y_true, y_pred),
            "precision": lambda: precision_score(y_true, y_pred),
            "recall": lambda: recall_score(y_true, y_pred),
            "auroc": lambda: roc_auc_score(y_true, y_score),
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
