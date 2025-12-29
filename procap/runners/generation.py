"""
Generation runners for ProCap benchmark.

Supports:
- Text generation (function description)
- Sequence generation (protein design)
"""

from typing import Any, Dict, List

import numpy as np

from procap.runners.base import BaseRunner
from procap.core.base_model import ModelOutput


class TextGenerationRunner(BaseRunner):
    """
    Runner for text generation tasks.

    Used for tasks like protein function description generation.

    Note: This runner expects generative models (like ProLLaMA, GPT-2)
    that can produce text output.
    """

    def _predict_batch(self, batch: List[Dict]) -> List[str]:
        """Generate text descriptions for batch."""
        sequences = [r[self.task.input_fields[0]] for r in batch]
        output: ModelOutput = self.model.predict_batch(sequences)

        if output.generated_text is not None:
            return output.generated_text

        # Fallback: return placeholder text
        return ["" for _ in batch]

    def _extract_targets(self, batch: List[Dict]) -> List[str]:
        """Extract reference text."""
        return [str(r.get(self.task.target_field, "")) for r in batch]

    def _compute_metrics(
        self,
        predictions: List[str],
        targets: List[str],
        metadata: List[Dict],
    ) -> Dict[str, float]:
        """Compute text generation metrics."""
        from procap.metrics.generation import (
            rouge_l,
            bert_score,
            novelty_50,
        )

        # Filter out empty predictions
        valid_pairs = [
            (p, t) for p, t in zip(predictions, targets) if p and t
        ]

        if not valid_pairs:
            return {m: 0.0 for m in self.task.metrics}

        valid_preds, valid_targets = zip(*valid_pairs)
        valid_preds = list(valid_preds)
        valid_targets = list(valid_targets)

        metric_funcs = {
            "rougeL": lambda: rouge_l(valid_targets, valid_preds),
            "bertscore": lambda: bert_score(valid_targets, valid_preds),
            "novelty_50": lambda: novelty_50(valid_targets, valid_preds),
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


class SequenceGenerationRunner(BaseRunner):
    """
    Runner for protein sequence generation tasks.

    Used for tasks like de novo protein design or inverse folding.
    """

    def _predict_batch(self, batch: List[Dict]) -> List[str]:
        """Generate protein sequences for batch."""
        # For sequence generation, we might have conditioning inputs
        conditions = []
        for r in batch:
            if "design_condition" in r:
                conditions.append(r["design_condition"])
            elif len(self.task.input_fields) > 0:
                conditions.append(r.get(self.task.input_fields[0], ""))
            else:
                conditions.append("")

        output: ModelOutput = self.model.predict_batch(conditions)

        if output.generated_text is not None:
            return output.generated_text

        return ["" for _ in batch]

    def _extract_targets(self, batch: List[Dict]) -> List[str]:
        """Extract reference sequences."""
        return [str(r.get(self.task.target_field, "")) for r in batch]

    def _compute_metrics(
        self,
        predictions: List[str],
        targets: List[str],
        metadata: List[Dict],
    ) -> Dict[str, float]:
        """Compute sequence generation metrics."""
        # Filter valid pairs
        valid_pairs = [
            (p, t) for p, t in zip(predictions, targets) if p
        ]

        if not valid_pairs:
            return {m: 0.0 for m in self.task.metrics}

        valid_preds, valid_targets = zip(*valid_pairs)
        valid_preds = list(valid_preds)
        valid_targets = list(valid_targets)

        results = {}

        # Sequence identity
        if "seq_identity" in self.task.metrics:
            identities = []
            for pred, target in zip(valid_preds, valid_targets):
                if target:
                    matches = sum(a == b for a, b in zip(pred, target))
                    identities.append(matches / max(len(pred), len(target), 1))
            results["seq_identity"] = np.mean(identities) if identities else 0.0

        # Diversity
        if "diversity" in self.task.metrics:
            if len(valid_preds) > 1:
                from itertools import combinations

                distances = []
                for s1, s2 in combinations(valid_preds, 2):
                    # Simple edit distance ratio
                    matches = sum(a == b for a, b in zip(s1, s2))
                    max_len = max(len(s1), len(s2), 1)
                    distances.append(1 - matches / max_len)
                results["diversity"] = np.mean(distances)
            else:
                results["diversity"] = 0.0

        return results
