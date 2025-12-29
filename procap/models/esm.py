"""
ESM-2 Model Adapter for ProCap Benchmark.

ESM (Evolutionary Scale Modeling) is a family of protein language models from Meta AI.
ESM-2 uses a BERT-like architecture trained on millions of protein sequences.

Key characteristics:
- Uses custom ESM tokenizer (NOT space-separated - contiguous sequences)
- Returns per-residue embeddings [batch, seq_len, hidden]
- Has special tokens: <cls> (position 0), <eos>, <pad>
- Supports various model sizes from 8M to 15B parameters

Model variants:
- esm2_t6_8M_UR50D: 6 layers, 8M params (fast, lightweight)
- esm2_t12_35M_UR50D: 12 layers, 35M params
- esm2_t30_150M_UR50D: 30 layers, 150M params
- esm2_t33_650M_UR50D: 33 layers, 650M params (recommended)
- esm2_t36_3B_UR50D: 36 layers, 3B params
- esm2_t48_15B_UR50D: 48 layers, 15B params (requires multiple GPUs)

Usage:
    model = ESMAdapter(variant="esm2_t33_650M")
    model.load()
    embeddings = model.get_embeddings(["MLKFV", "ACDEFG"], pooling="mean")
"""

from typing import List, Literal, Optional

import torch
from transformers import AutoTokenizer, AutoModel

from procap.core.base_model import BaseProteinModel, ModelOutput, TokenizerOutput


class ESMAdapter(BaseProteinModel):
    """
    Adapter for ESM-1b and ESM-2 models.

    IMPORTANT: ESM uses contiguous amino acid sequences (no spaces).
    Input: "MLKFV" NOT "M L K F V"
    """

    MODEL_VARIANTS = {
        # ESM-2 models (recommended)
        "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
        "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
        "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
        "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
        "esm2_t36_3B": "facebook/esm2_t36_3B_UR50D",
        "esm2_t48_15B": "facebook/esm2_t48_15B_UR50D",
        # ESM-1b model
        "esm1b_t33_650M": "facebook/esm1b_t33_650M_UR50S",
        # ESM-1v models (optimized for variant effect prediction, 5 model ensemble)
        "esm1v_t33_650M_1": "facebook/esm1v_t33_650M_UR90S_1",
        "esm1v_t33_650M_2": "facebook/esm1v_t33_650M_UR90S_2",
        "esm1v_t33_650M_3": "facebook/esm1v_t33_650M_UR90S_3",
        "esm1v_t33_650M_4": "facebook/esm1v_t33_650M_UR90S_4",
        "esm1v_t33_650M_5": "facebook/esm1v_t33_650M_UR90S_5",
    }

    def __init__(
        self,
        variant: str = "esm2_t33_650M",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 16,
        max_sequence_length: int = 1022,  # ESM max is 1024 including special tokens
        **kwargs,
    ):
        """
        Initialize ESM adapter.

        Args:
            variant: Model variant (e.g., "esm2_t33_650M")
            device: Device to use (None for auto-detect)
            dtype: Data type for model weights
            max_batch_size: Maximum batch size (lower for larger models)
            max_sequence_length: Maximum sequence length (ESM supports up to 1022 residues)
        """
        name = f"ESM-{variant}"
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
            # Allow direct repo ID
            self.repo_id = variant

    def load(self) -> None:
        """Load ESM model and tokenizer from HuggingFace."""
        print(f"Loading {self.name} from {self.repo_id}...")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.repo_id)

        # Load model
        self._model = AutoModel.from_pretrained(
            self.repo_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self._model.eval()

        # Set model properties
        self.hidden_size = self._model.config.hidden_size
        self.num_layers = self._model.config.num_hidden_layers
        self._is_loaded = True

        print(f"Loaded {self.name}: hidden_size={self.hidden_size}, layers={self.num_layers}")

    def _preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess sequence for ESM tokenization.

        ESM uses contiguous sequences (no spaces).
        Removes spaces, converts to uppercase.
        """
        # Remove spaces and convert to uppercase
        sequence = sequence.upper().replace(" ", "")

        # Truncate if too long (leave room for <cls> and <eos>)
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[: self.max_sequence_length]

        return sequence

    def tokenize(self, sequences: List[str]) -> TokenizerOutput:
        """
        Tokenize sequences for ESM model.

        ESM uses contiguous amino acid sequences (no spaces).
        """
        # Preprocess all sequences
        processed_seqs = [self._preprocess_sequence(seq) for seq in sequences]

        # Tokenize
        encoded = self._tokenizer(
            processed_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length + 2,  # +2 for special tokens
            return_tensors="pt",
        )

        return TokenizerOutput(
            input_ids=encoded["input_ids"].to(self.device),
            attention_mask=encoded["attention_mask"].to(self.device),
            sequence_lengths=[len(s) for s in processed_seqs],
        )

    def get_embeddings(
        self,
        sequences: List[str],
        pooling: Literal["mean", "cls", "max", "per_residue"] = "mean",
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Extract embeddings from ESM model.

        Args:
            sequences: List of amino acid sequences (contiguous, no spaces)
            pooling:
                - "mean": Average over all residue positions (excluding special tokens)
                - "cls": Use <cls> token embedding (position 0)
                - "max": Max pooling over residues
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
        if layer != -1 and hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[layer]
        else:
            hidden_states = outputs.last_hidden_state

        if pooling == "per_residue":
            return hidden_states

        elif pooling == "cls":
            # ESM puts <cls> at position 0
            return hidden_states[:, 0, :]

        elif pooling == "mean":
            # Mean pooling excluding special tokens (first and last positions)
            # Create mask that excludes <cls> (pos 0) and <eos>/<pad>
            mask = tokens.attention_mask.unsqueeze(-1).float()

            # Zero out <cls> position
            mask[:, 0, :] = 0

            # Zero out <eos> position for each sequence
            if tokens.sequence_lengths:
                for i, length in enumerate(tokens.sequence_lengths):
                    if length + 1 < mask.shape[1]:
                        mask[i, length + 1, :] = 0

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
