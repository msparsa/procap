"""
Task-specific heads for protein language models.

These heads are attached to PLM encoders to perform specific tasks:
- MultiLabelClassificationHead: For GO term prediction (multi-label)
- BinaryClassificationHead: For binary classification tasks
- RegressionHead: For variant effect prediction, binding affinity, etc.
"""

from typing import Optional

import torch
import torch.nn as nn


class MultiLabelClassificationHead(nn.Module):
    """
    Multi-label classification head for GO term prediction.

    Takes sequence embeddings and predicts multiple GO terms.

    Architecture:
        Linear -> ReLU -> Dropout -> Linear

    Usage:
        head = MultiLabelClassificationHead(hidden_size=768, num_labels=5000)
        logits = head(embeddings)  # [batch, num_labels]
        probs = torch.sigmoid(logits)  # For multi-label
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize multi-label classification head.

        Args:
            hidden_size: Input embedding dimension
            num_labels: Number of labels (GO terms)
            dropout: Dropout probability
            hidden_dim: Hidden layer dimension (default: hidden_size // 2)
        """
        super().__init__()

        hidden_dim = hidden_dim or hidden_size // 2

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

        self.num_labels = num_labels
        self.hidden_size = hidden_size

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: [batch_size, hidden_size]

        Returns:
            logits: [batch_size, num_labels]
        """
        return self.classifier(embeddings)

    def get_probabilities(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get probability scores using sigmoid."""
        logits = self.forward(embeddings)
        return torch.sigmoid(logits)


class BinaryClassificationHead(nn.Module):
    """
    Binary classification head.

    For tasks like protein-protein interaction prediction.

    Architecture:
        Dropout -> Linear

    Usage:
        head = BinaryClassificationHead(hidden_size=768)
        logits = head(embeddings)  # [batch]
        probs = torch.sigmoid(logits)
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        """
        Initialize binary classification head.

        Args:
            hidden_size: Input embedding dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.hidden_size = hidden_size

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: [batch_size, hidden_size]

        Returns:
            logits: [batch_size]
        """
        return self.classifier(embeddings).squeeze(-1)

    def get_probabilities(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get probability scores using sigmoid."""
        logits = self.forward(embeddings)
        return torch.sigmoid(logits)


class RegressionHead(nn.Module):
    """
    Regression head for continuous value prediction.

    For tasks like:
    - Variant effect prediction (DDG)
    - Binding affinity prediction
    - Stability prediction

    Architecture:
        Linear -> ReLU -> Dropout -> Linear

    Usage:
        head = RegressionHead(hidden_size=768)
        predictions = head(embeddings)  # [batch, 1]
    """

    def __init__(
        self,
        hidden_size: int,
        num_outputs: int = 1,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize regression head.

        Args:
            hidden_size: Input embedding dimension
            num_outputs: Number of output values (default 1)
            dropout: Dropout probability
            hidden_dim: Hidden layer dimension (default: hidden_size // 2)
        """
        super().__init__()

        hidden_dim = hidden_dim or hidden_size // 2

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

        self.num_outputs = num_outputs
        self.hidden_size = hidden_size

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: [batch_size, hidden_size]

        Returns:
            predictions: [batch_size, num_outputs]
        """
        return self.regressor(embeddings)


class SequenceClassificationHead(nn.Module):
    """
    Per-residue classification head.

    For tasks like secondary structure prediction where each residue
    gets a class label.

    Architecture:
        Dropout -> Linear

    Usage:
        head = SequenceClassificationHead(hidden_size=768, num_classes=3)  # H, E, C
        logits = head(hidden_states)  # [batch, seq_len, num_classes]
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        """
        Initialize sequence classification head.

        Args:
            hidden_size: Input embedding dimension per position
            num_classes: Number of classes per position
            dropout: Dropout probability
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        self.num_classes = num_classes
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            logits: [batch_size, seq_len, num_classes]
        """
        return self.classifier(hidden_states)


class DeltaDeltaGHead(nn.Module):
    """
    Specialized head for stability change prediction (DDG).

    Takes wild-type and mutant embeddings and predicts the change
    in folding free energy (DDG).

    Architecture:
        Concatenate(WT, Mutant) -> Linear -> ReLU -> Dropout -> Linear

    Usage:
        head = DeltaDeltaGHead(hidden_size=768)
        ddg = head(wt_embedding, mut_embedding)  # [batch]
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        """
        Initialize DDG prediction head.

        Args:
            hidden_size: Input embedding dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Process concatenated WT and mutant embeddings
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.hidden_size = hidden_size

    def forward(
        self,
        wt_embedding: torch.Tensor,
        mut_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            wt_embedding: Wild-type embeddings [batch_size, hidden_size]
            mut_embedding: Mutant embeddings [batch_size, hidden_size]

        Returns:
            ddg: Predicted DDG values [batch_size]
        """
        combined = torch.cat([wt_embedding, mut_embedding], dim=-1)
        return self.regressor(combined).squeeze(-1)
