"""
Base model class and data structures for all protein language model adapters.

Each PLM (ESM-2, ProtBERT, OntoProtein, etc.) should inherit from BaseProteinModel
and implement the required methods with their specific tokenization and inference logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np


@dataclass
class TokenizerOutput:
    """Unified tokenizer output structure."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    special_tokens_mask: Optional[torch.Tensor] = None
    sequence_lengths: Optional[List[int]] = None

    def to(self, device: Union[str, torch.device]) -> "TokenizerOutput":
        """Move tensors to specified device."""
        return TokenizerOutput(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            special_tokens_mask=self.special_tokens_mask.to(device) if self.special_tokens_mask is not None else None,
            sequence_lengths=self.sequence_lengths,
        )


@dataclass
class ModelOutput:
    """Unified output structure for all model types."""

    # Embeddings from the model
    embeddings: Optional[torch.Tensor] = None  # [batch, hidden] or [batch, seq_len, hidden]

    # For classification tasks
    logits: Optional[torch.Tensor] = None  # [batch, num_classes]
    probabilities: Optional[torch.Tensor] = None  # [batch, num_classes]

    # For regression tasks
    predictions: Optional[torch.Tensor] = None  # [batch, 1] or [batch]

    # For generation tasks
    generated_ids: Optional[torch.Tensor] = None  # [batch, seq_len]
    generated_text: Optional[List[str]] = None

    # Attention weights if requested
    attention_weights: Optional[torch.Tensor] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_numpy(self) -> "ModelOutput":
        """Convert all tensors to numpy arrays."""
        return ModelOutput(
            embeddings=self.embeddings.cpu().numpy() if self.embeddings is not None else None,
            logits=self.logits.cpu().numpy() if self.logits is not None else None,
            probabilities=self.probabilities.cpu().numpy() if self.probabilities is not None else None,
            predictions=self.predictions.cpu().numpy() if self.predictions is not None else None,
            generated_ids=self.generated_ids.cpu().numpy() if self.generated_ids is not None else None,
            generated_text=self.generated_text,
            attention_weights=self.attention_weights.cpu().numpy() if self.attention_weights is not None else None,
            metadata=self.metadata,
        )


class BaseProteinModel(ABC):
    """
    Abstract base class for all protein language model adapters.

    Each model family (ESM, ProtBERT, OntoProtein, etc.) should implement this interface
    with their specific tokenization and inference logic.

    Key methods to implement:
    - load(): Load model weights and tokenizer
    - tokenize(): Convert sequences to model-specific token format
    - get_embeddings(): Extract sequence embeddings with pooling
    - _predict_single_batch(): Model-specific prediction logic

    Example usage:
        model = ESMAdapter(variant="esm2_t33_650M")
        model.load()
        embeddings = model.get_embeddings(["MLKFV", "ACDEFG"])
    """

    def __init__(
        self,
        name: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 32,
        max_sequence_length: int = 1024,
    ):
        """
        Initialize base model.

        Args:
            name: Model identifier
            device: Device to use ("cuda", "cpu", or None for auto)
            dtype: Data type for model weights
            max_batch_size: Maximum batch size for inference
            max_sequence_length: Maximum sequence length (will truncate longer sequences)
        """
        self.name = name
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length

        # To be set by subclasses
        self._model: Optional[nn.Module] = None
        self._tokenizer = None
        self._head: Optional[nn.Module] = None
        self.hidden_size: int = 0
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights and tokenizer.

        Should set:
        - self._model
        - self._tokenizer
        - self.hidden_size
        - self._is_loaded = True
        """
        pass

    @abstractmethod
    def tokenize(self, sequences: List[str]) -> TokenizerOutput:
        """
        Tokenize protein sequences according to model-specific requirements.

        IMPORTANT: Different models have different requirements:
        - ESM-2: Contiguous sequences ("MLKFV")
        - ProtBERT: Space-separated ("M L K F V")
        - OntoProtein: Space-separated ("M L K F V")

        Args:
            sequences: List of amino acid sequences

        Returns:
            TokenizerOutput with input_ids and attention_mask
        """
        pass

    @abstractmethod
    def get_embeddings(
        self,
        sequences: List[str],
        pooling: Literal["mean", "cls", "max", "per_residue"] = "mean",
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Extract embeddings from sequences.

        Args:
            sequences: List of amino acid sequences
            pooling: Pooling strategy
                - "mean": Average over all residue positions
                - "cls": Use [CLS] token embedding
                - "max": Max pooling over residues
                - "per_residue": Return full [batch, seq_len, hidden] tensor
            layer: Which transformer layer to extract from (-1 = last)

        Returns:
            Embeddings tensor of shape [batch, hidden] or [batch, seq_len, hidden]
        """
        pass

    def attach_head(self, head: nn.Module) -> None:
        """
        Attach a task-specific head to the model.

        Args:
            head: Neural network head for classification/regression
        """
        self._head = head.to(self.device)

    def detach_head(self) -> Optional[nn.Module]:
        """Remove and return the attached head."""
        head = self._head
        self._head = None
        return head

    @torch.inference_mode()
    def predict_batch(
        self,
        sequences: List[str],
        return_embeddings: bool = False,
    ) -> ModelOutput:
        """
        Run inference on a batch of sequences.

        Handles automatic batching if input exceeds max_batch_size.

        Args:
            sequences: List of amino acid sequences
            return_embeddings: Whether to include embeddings in output

        Returns:
            ModelOutput with predictions
        """
        if not self._is_loaded:
            raise RuntimeError(f"Model {self.name} not loaded. Call load() first.")

        if len(sequences) <= self.max_batch_size:
            return self._predict_single_batch(sequences, return_embeddings)

        # Split into sub-batches
        outputs = []
        for i in range(0, len(sequences), self.max_batch_size):
            batch = sequences[i : i + self.max_batch_size]
            outputs.append(self._predict_single_batch(batch, return_embeddings))

        return self._merge_outputs(outputs)

    @abstractmethod
    def _predict_single_batch(
        self,
        sequences: List[str],
        return_embeddings: bool,
    ) -> ModelOutput:
        """
        Implementation-specific prediction logic for a single batch.

        Args:
            sequences: List of amino acid sequences (len <= max_batch_size)
            return_embeddings: Whether to include embeddings in output

        Returns:
            ModelOutput with predictions
        """
        pass

    def _merge_outputs(self, outputs: List[ModelOutput]) -> ModelOutput:
        """Merge multiple batch outputs into one."""
        if not outputs:
            return ModelOutput()

        merged = ModelOutput(metadata={"model": self.name, "num_batches": len(outputs)})

        # Merge embeddings
        if outputs[0].embeddings is not None:
            merged.embeddings = torch.cat([o.embeddings for o in outputs], dim=0)

        # Merge logits
        if outputs[0].logits is not None:
            merged.logits = torch.cat([o.logits for o in outputs], dim=0)

        # Merge probabilities
        if outputs[0].probabilities is not None:
            merged.probabilities = torch.cat([o.probabilities for o in outputs], dim=0)

        # Merge predictions
        if outputs[0].predictions is not None:
            merged.predictions = torch.cat([o.predictions for o in outputs], dim=0)

        # Merge generated text
        if outputs[0].generated_text is not None:
            merged.generated_text = []
            for o in outputs:
                merged.generated_text.extend(o.generated_text)

        return merged

    @contextmanager
    def memory_efficient_mode(self):
        """
        Context manager for memory-efficient inference using fp16.

        Example:
            with model.memory_efficient_mode():
                embeddings = model.get_embeddings(sequences)
        """
        if self._model is None:
            yield
            return

        original_dtype = self.dtype
        try:
            if self.device.type == "cuda" and self.dtype == torch.float32:
                self._model = self._model.half()
                self.dtype = torch.float16
            yield
        finally:
            if original_dtype == torch.float32 and self.dtype == torch.float16:
                self._model = self._model.float()
                self.dtype = original_dtype

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"device={self.device}, "
            f"loaded={self._is_loaded})"
        )
