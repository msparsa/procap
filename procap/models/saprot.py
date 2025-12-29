"""
SaProt Model Adapter for ProCap Benchmark.

SaProt is a structure-aware protein language model from ICLR 2024.
It uses 3Di structure tokens from Foldseek combined with sequence tokens.

Key characteristics:
- Uses combined sequence + structure tokens (e.g., "MaLbKcFdVe")
- Lowercase letters represent 3Di structure tokens
- Can work with sequence-only input (structure tokens masked)
- Based on ESM-2 architecture with structure awareness

Model variants:
- SaProt_650M_AF2: Trained on AlphaFold2 structures
- SaProt_650M_PDB: Trained on PDB structures

Usage:
    model = SaProtAdapter(variant="saprot_650m_af2")
    model.load()
    # With structure tokens
    embeddings = model.get_embeddings(["MaLbKcFdVe"], pooling="mean")
    # Sequence-only (auto-adds dummy structure tokens)
    embeddings = model.get_embeddings(["MLKFV"], pooling="mean")
"""

from typing import List, Literal, Optional

import torch
from transformers import AutoTokenizer, AutoModel, EsmTokenizer

from procap.core.base_model import BaseProteinModel, ModelOutput, TokenizerOutput


class SaProtAdapter(BaseProteinModel):
    """
    Adapter for SaProt structure-aware protein language model.

    SaProt combines amino acid sequence with 3Di structure tokens.
    Input format: "MaLbKcFdVe" where uppercase = amino acid, lowercase = 3Di token

    For sequence-only input, the adapter automatically adds placeholder structure tokens.
    """

    MODEL_VARIANTS = {
        "saprot_650m_af2": "westlake-repl/SaProt_650M_AF2",
        "saprot_650m_pdb": "westlake-repl/SaProt_650M_PDB",
    }

    # Default 3Di token to use when structure is not provided
    DEFAULT_STRUCTURE_TOKEN = "d"  # Most common 3Di token

    def __init__(
        self,
        variant: str = "saprot_650m_af2",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 8,
        max_sequence_length: int = 1022,
        **kwargs,
    ):
        """
        Initialize SaProt adapter.

        Args:
            variant: Model variant (e.g., "saprot_650m_af2")
            device: Device to use (None for auto-detect)
            dtype: Data type for model weights
            max_batch_size: Maximum batch size
            max_sequence_length: Maximum sequence length
        """
        name = f"SaProt-{variant}"
        super().__init__(
            name=name,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
        )

        self.variant = variant
        if variant in self.MODEL_VARIANTS:
            self.repo_id = self.MODEL_VARIANTS[variant]
        else:
            self.repo_id = variant

    def load(self) -> None:
        """Load SaProt model and tokenizer from HuggingFace."""
        print(f"Loading {self.name} from {self.repo_id}...")

        # Load tokenizer - SaProt uses ESM tokenizer
        self._tokenizer = EsmTokenizer.from_pretrained(
            self.repo_id,
            trust_remote_code=True,
        )

        # Load model
        self._model = AutoModel.from_pretrained(
            self.repo_id,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self._model.eval()

        # Set model properties
        self.hidden_size = self._model.config.hidden_size
        self.num_layers = self._model.config.num_hidden_layers
        self._is_loaded = True

        print(f"Loaded {self.name}: hidden_size={self.hidden_size}, layers={self.num_layers}")

    def _has_structure_tokens(self, sequence: str) -> bool:
        """Check if sequence contains 3Di structure tokens (lowercase letters)."""
        return any(c.islower() for c in sequence)

    def _add_structure_tokens(self, sequence: str) -> str:
        """
        Add default structure tokens to a sequence-only input.

        Converts "MLKFV" -> "MdLdKdFdVd" using default 3Di token.
        """
        result = []
        for aa in sequence.upper():
            if aa.isalpha():
                result.append(aa + self.DEFAULT_STRUCTURE_TOKEN)
            else:
                result.append(aa)
        return "".join(result)

    def _preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess sequence for SaProt tokenization.

        If sequence doesn't have structure tokens, adds default ones.
        """
        # Remove extra spaces
        sequence = sequence.strip()

        # If no structure tokens, add default ones
        if not self._has_structure_tokens(sequence):
            sequence = self._add_structure_tokens(sequence)

        # Truncate if too long (considering interleaved format doubles length)
        effective_length = len([c for c in sequence if c.isupper()])
        if effective_length > self.max_sequence_length:
            # Truncate while preserving pairs
            chars = list(sequence)
            kept = []
            count = 0
            for i, c in enumerate(chars):
                if c.isupper():
                    count += 1
                    if count > self.max_sequence_length:
                        break
                kept.append(c)
                # Include the following structure token if present
                if c.isupper() and i + 1 < len(chars) and chars[i + 1].islower():
                    kept.append(chars[i + 1])
            sequence = "".join(kept)

        return sequence

    def tokenize(self, sequences: List[str]) -> TokenizerOutput:
        """
        Tokenize sequences for SaProt model.

        Handles both sequence+structure format and sequence-only format.
        """
        # Preprocess all sequences
        processed_seqs = [self._preprocess_sequence(seq) for seq in sequences]

        # Tokenize
        encoded = self._tokenizer(
            processed_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length * 2 + 2,  # *2 for interleaved, +2 for special tokens
            return_tensors="pt",
        )

        # Calculate effective sequence lengths (number of amino acids)
        sequence_lengths = [
            len([c for c in seq if c.isupper()]) for seq in processed_seqs
        ]

        return TokenizerOutput(
            input_ids=encoded["input_ids"].to(self.device),
            attention_mask=encoded["attention_mask"].to(self.device),
            sequence_lengths=sequence_lengths,
        )

    def get_embeddings(
        self,
        sequences: List[str],
        pooling: Literal["mean", "cls", "max", "per_residue"] = "mean",
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Extract embeddings from SaProt model.

        Args:
            sequences: List of sequences (with or without structure tokens)
            pooling:
                - "mean": Average over all positions (excluding special tokens)
                - "cls": Use <cls> token embedding (position 0)
                - "max": Max pooling over positions
                - "per_residue": Return full [batch, seq_len, hidden] tensor
            layer: Which transformer layer to extract from (-1 = last)

        Returns:
            Embeddings tensor
        """
        if not self._is_loaded:
            raise RuntimeError(f"Model {self.name} not loaded. Call load() first.")

        tokens = self.tokenize(sequences)

        with torch.inference_mode():
            outputs = self._model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=(layer != -1),
            )

        # Get hidden states from specified layer
        if layer != -1 and hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden_states = outputs.hidden_states[layer]
        else:
            hidden_states = outputs.last_hidden_state

        if pooling == "per_residue":
            return hidden_states

        elif pooling == "cls":
            # SaProt puts <cls> at position 0
            return hidden_states[:, 0, :]

        elif pooling == "mean":
            # Mean pooling excluding special tokens
            mask = tokens.attention_mask.unsqueeze(-1).float()
            # Zero out <cls> position
            mask[:, 0, :] = 0
            # Compute mean
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            return summed / counts

        elif pooling == "max":
            # Max pooling with attention mask
            mask = tokens.attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states_masked = hidden_states.clone()
            hidden_states_masked[mask == 0] = float("-inf")
            return hidden_states_masked.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

    def _predict_single_batch(
        self,
        sequences: List[str],
        return_embeddings: bool = False,
    ) -> ModelOutput:
        """Run prediction on a single batch."""
        embeddings = self.get_embeddings(sequences, pooling="mean")

        output = ModelOutput(metadata={"model": self.name, "variant": self.variant})

        if return_embeddings:
            output.embeddings = embeddings

        # Apply head if attached
        if self._head is not None:
            head_output = self._head(embeddings)
            if isinstance(head_output, tuple):
                output.logits = head_output[0]
                if len(head_output) > 1:
                    output.predictions = head_output[1]
            else:
                output.logits = head_output

        return output

    @classmethod
    def list_variants(cls) -> List[str]:
        """List available model variants."""
        return list(cls.MODEL_VARIANTS.keys())
