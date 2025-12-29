"""
Generation metrics for ProCap benchmark.

Supports text generation metrics like ROUGE, BERTScore, and novelty.
"""

from typing import List
import numpy as np


def rouge_l(references: List[str], predictions: List[str]) -> float:
    """
    Compute ROUGE-L F1 score.

    Args:
        references: List of reference texts
        predictions: List of generated texts

    Returns:
        Average ROUGE-L F1 score
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        scores = []
        for ref, pred in zip(references, predictions):
            if ref and pred:
                score = scorer.score(ref, pred)
                scores.append(score["rougeL"].fmeasure)

        return float(np.mean(scores)) if scores else 0.0

    except ImportError:
        print("Warning: rouge_score not installed. Using simple fallback.")
        return _simple_rouge_l(references, predictions)


def _simple_rouge_l(references: List[str], predictions: List[str]) -> float:
    """Simple ROUGE-L fallback using LCS."""

    def lcs_length(s1: str, s2: str) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    scores = []
    for ref, pred in zip(references, predictions):
        if ref and pred:
            ref_tokens = ref.lower().split()
            pred_tokens = pred.lower().split()
            ref_str = " ".join(ref_tokens)
            pred_str = " ".join(pred_tokens)

            lcs = lcs_length(ref_str, pred_str)
            precision = lcs / len(pred_str) if pred_str else 0
            recall = lcs / len(ref_str) if ref_str else 0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                scores.append(f1)

    return float(np.mean(scores)) if scores else 0.0


def bert_score(references: List[str], predictions: List[str]) -> float:
    """
    Compute BERTScore F1.

    Args:
        references: List of reference texts
        predictions: List of generated texts

    Returns:
        Average BERTScore F1
    """
    try:
        from bert_score import score as bert_score_fn

        if not references or not predictions:
            return 0.0

        # Filter out empty strings
        valid_pairs = [(r, p) for r, p in zip(references, predictions) if r and p]
        if not valid_pairs:
            return 0.0

        refs, preds = zip(*valid_pairs)

        # Compute BERTScore (using smaller model for efficiency)
        P, R, F1 = bert_score_fn(
            list(preds),
            list(refs),
            model_type="distilbert-base-uncased",
            verbose=False,
        )

        return float(F1.mean())

    except ImportError:
        print("Warning: bert_score not installed. Returning 0.")
        return 0.0
    except Exception as e:
        print(f"Warning: BERTScore failed: {e}")
        return 0.0


def novelty_50(references: List[str], predictions: List[str]) -> float:
    """
    Compute fraction of predictions with >50% difference from references.

    Uses Levenshtein distance to measure text novelty.

    Args:
        references: List of reference texts
        predictions: List of generated texts

    Returns:
        Fraction of novel predictions (>50% different)
    """

    def levenshtein_ratio(s1: str, s2: str) -> float:
        """Compute Levenshtein similarity ratio (0-1, higher = more similar)."""
        if not s1 or not s2:
            return 0.0

        len1, len2 = len(s1), len(s2)

        # Use dynamic programming for edit distance
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        previous_row = list(range(len1 + 1))
        for i, c2 in enumerate(s2):
            current_row = [i + 1]
            for j, c1 in enumerate(s1):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        distance = previous_row[-1]
        max_len = max(len1, len2)
        return 1 - (distance / max_len)

    novel_count = 0
    total = 0

    for ref, pred in zip(references, predictions):
        if pred:
            total += 1
            # Check similarity to reference
            if ref:
                similarity = levenshtein_ratio(ref.lower(), pred.lower())
                # >50% different means <50% similar
                if similarity < 0.5:
                    novel_count += 1
            else:
                # No reference = novel
                novel_count += 1

    return float(novel_count / total) if total > 0 else 0.0


def bleu_score(references: List[str], predictions: List[str]) -> float:
    """
    Compute BLEU score.

    Args:
        references: List of reference texts
        predictions: List of generated texts

    Returns:
        BLEU score
    """
    try:
        import sacrebleu

        if not references or not predictions:
            return 0.0

        # Filter valid pairs
        valid_pairs = [(r, p) for r, p in zip(references, predictions) if r and p]
        if not valid_pairs:
            return 0.0

        refs, preds = zip(*valid_pairs)

        # sacrebleu expects list of references for each prediction
        refs_list = [[r] for r in refs]

        bleu = sacrebleu.corpus_bleu(list(preds), list(zip(*refs_list)))
        return float(bleu.score / 100)  # Normalize to 0-1

    except ImportError:
        print("Warning: sacrebleu not installed. Returning 0.")
        return 0.0
    except Exception as e:
        print(f"Warning: BLEU failed: {e}")
        return 0.0
