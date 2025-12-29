"""
ProtBERT Model Adapter for ProCap Benchmark.

ProtBERT is a BERT model pre-trained on protein sequences from UniRef and BFD.
It was developed by Rostlab and is widely used for protein sequence analysis.

Key characteristics:
- CRITICAL: Requires space-separated amino acids ("M L K F V")
- Uses standard BertTokenizer with do_lower_case=False
- Rare amino acids (U, Z, O, B) should be replaced with X
- Maximum sequence length: 512 tokens (including special tokens)
- Uses [CLS] and [SEP] special tokens

Usage:
    model = ProtBERTAdapter()
    model.load()
    # Note: sequences will be automatically converted to space-separated format
    embeddings = model.get_embeddings(["MLKFV", "ACDEFG"], pooling="mean")
"""

import re
from typing import List, Literal, Optional

import torch
from transformers import BertModel, BertTokenizer

from procap.core.base_model import BaseProteinModel, ModelOutput, TokenizerOutput


class ProtBERTAdapter(BaseProteinModel):
    """
    Adapter for ProtBERT (Rostlab/prot_bert).

    CRITICAL: ProtBERT requires space-separated amino acids.
    Input must be: "M L K F V" NOT "MLKFV"

    This adapter automatically handles the conversion.
    """

    MODEL_VARIANTS = {
        "prot_bert": "Rostlab/prot_bert",
        "prot_bert_bfd": "Rostlab/prot_bert_bfd",
    }

    def __init__(
        self,
        variant: str = "prot_bert",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 32,
        max_sequence_length: int = 510,  # 512 - 2 for [CLS] and [SEP]
        **kwargs,
    ):
        """
        Initialize ProtBERT adapter.

        Args:
            variant: Model variant ("prot_bert" or "prot_bert_bfd")
            device: Device to use (None for auto-detect)
            dtype: Data type for model weights
            max_batch_size: Maximum batch size
            max_sequence_length: Maximum sequence length (max 510 residues)
        """
        name = f"ProtBERT-{variant}"
        super().__init__(
            name=name,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
            max_sequence_length=min(max_sequence_length, 510),  # Cap at 510
        )

        self.variant = variant
        if variant in self.MODEL_VARIANTS:
            self.repo_id = self.MODEL_VARIANTS[variant]
        else:
            self.repo_id = variant

    def load(self) -> None:
        """Load ProtBERT model and tokenizer from HuggingFace."""
        print(f"Loading {self.name} from {self.repo_id}...")

        # Load tokenizer with do_lower_case=False (important for amino acids)
        self._tokenizer = BertTokenizer.from_pretrained(
            self.repo_id,
            do_lower_case=False,
        )

        # Load model
        self._model = BertModel.from_pretrained(
            self.repo_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self._model.eval()

        # Set model properties
        self.hidden_size = self._model.config.hidden_size
        self.num_layers = self._model.config.num_hidden_layers
        self._is_loaded = True

        print(f"Loaded {self.name}: hidden_size={self.hidden_size}, layers={self.num_layers}")

    @staticmethod
    def preprocess_sequence(sequence: str) -> str:
        """
        Preprocess sequence for ProtBERT tokenization.

        CRITICAL: Converts contiguous sequence to space-separated format.
        "MLKFV" -> "M L K F V"

        Also replaces rare amino acids (U, Z, O, B) with X.
        """
        # Remove existing spaces and convert to uppercase
        sequence = sequence.upper().replace(" ", "")

        # Replace rare amino acids with X
        # U = Selenocysteine, Z = Glutamic acid or Glutamine
        # O = Pyrrolysine, B = Aspartic acid or Asparagine
        sequence = re.sub(r"[UZOB]", "X", sequence)

        # Convert to space-separated format
        # "MLKFV" -> "M L K F V"
        return " ".join(list(sequence))

    def _preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess a single sequence for tokenization.

        Converts to space-separated format and truncates if needed.
        """
        # Convert to space-separated
        processed = self.preprocess_sequence(sequence)

        # Truncate if too long (in terms of residues, not tokens)
        residues = processed.split()
        if len(residues) > self.max_sequence_length:
            residues = residues[: self.max_sequence_length]
            processed = " ".join(residues)

        return processed

    def tokenize(self, sequences: List[str]) -> TokenizerOutput:
        """
        Tokenize sequences for ProtBERT model.

        Automatically converts to space-separated format if needed.
        """
        # Preprocess all sequences (converts to space-separated)
        processed_seqs = [self._preprocess_sequence(seq) for seq in sequences]

        # Tokenize
        encoded = self._tokenizer(
            processed_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length + 2,  # +2 for [CLS] and [SEP]
            return_tensors="pt",
        )

        # Calculate original sequence lengths (number of residues)
        sequence_lengths = [len(seq.split()) for seq in processed_seqs]

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
        Extract embeddings from ProtBERT model.

        Args:
            sequences: List of amino acid sequences (can be contiguous or space-separated)
            pooling:
                - "mean": Average over all residue positions (excluding special tokens)
                - "cls": Use [CLS] token embedding (position 0)
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
            # BERT [CLS] token is at position 0
            return hidden_states[:, 0, :]

        elif pooling == "mean":
            # Mean pooling excluding [CLS] (pos 0) and [SEP]/[PAD]
            mask = tokens.attention_mask.unsqueeze(-1).float()

            # Zero out [CLS] position
            mask[:, 0, :] = 0

            # Compute mean over valid positions
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
