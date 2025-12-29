"""
OntoProtein Model Adapter for ProCap Benchmark.

OntoProtein is a BERT-based model that incorporates Gene Ontology (GO) knowledge
into protein representations. It was developed by ZJU NLP and combines
sequence-level protein language modeling with structured knowledge from GO.

Key characteristics:
- BERT-based architecture (similar to ProtBERT)
- Requires space-separated amino acids ("M L K F V")
- Pre-trained with GO knowledge embeddings
- Particularly good for GO-related downstream tasks
- Uses [CLS] token for sequence-level predictions (GO knowledge encoded there)

Usage:
    model = OntoProteinAdapter()
    model.load()
    # Sequences are automatically converted to space-separated format
    embeddings = model.get_embeddings(["MLKFV", "ACDEFG"], pooling="cls")
"""

from typing import List, Literal, Optional

import torch
from transformers import BertModel, BertTokenizer

from procap.core.base_model import BaseProteinModel, ModelOutput, TokenizerOutput


class OntoProteinAdapter(BaseProteinModel):
    """
    Adapter for OntoProtein (zjunlp/OntoProtein).

    CRITICAL: OntoProtein requires space-separated amino acids (same as ProtBERT).
    Input must be: "M L K F V" NOT "MLKFV"

    This adapter automatically handles the conversion.

    The model incorporates Gene Ontology knowledge, so it's particularly
    well-suited for GO term prediction tasks.
    """

    MODEL_VARIANTS = {
        "ontoprotein": "zjunlp/OntoProtein",
    }

    def __init__(
        self,
        variant: str = "ontoprotein",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 32,
        max_sequence_length: int = 510,  # 512 - 2 for [CLS] and [SEP]
        **kwargs,
    ):
        """
        Initialize OntoProtein adapter.

        Args:
            variant: Model variant (default "ontoprotein")
            device: Device to use (None for auto-detect)
            dtype: Data type for model weights
            max_batch_size: Maximum batch size
            max_sequence_length: Maximum sequence length (max 510 residues)
        """
        name = "OntoProtein"
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
        """Load OntoProtein model and tokenizer from HuggingFace."""
        print(f"Loading {self.name} from {self.repo_id}...")

        # Load tokenizer with do_lower_case=False (important for amino acids)
        self._tokenizer = BertTokenizer.from_pretrained(
            self.repo_id,
            do_lower_case=False,
        )

        # Load model - OntoProtein has meta tensor issues (pooler weights are on meta device)
        # Use add_pooling_layer=False since we don't use the pooler (we use CLS token directly)
        self._model = BertModel.from_pretrained(
            self.repo_id,
            add_pooling_layer=False,  # Skip pooler to avoid meta tensor issues
        )

        # Convert dtype if needed
        if self.dtype == torch.float16:
            self._model = self._model.half()

        # Move to device
        self._model = self._model.to(self.device)
        self._model.eval()

        # Set model properties
        self.hidden_size = self._model.config.hidden_size
        self.num_layers = self._model.config.num_hidden_layers
        self._is_loaded = True

        print(f"Loaded {self.name}: hidden_size={self.hidden_size}, layers={self.num_layers}")

    @staticmethod
    def preprocess_sequence(sequence: str) -> str:
        """
        Preprocess sequence for OntoProtein tokenization.

        Converts contiguous sequence to space-separated format.
        "MLKFV" -> "M L K F V"
        """
        # Remove existing spaces and convert to uppercase
        sequence = sequence.upper().replace(" ", "")

        # Convert to space-separated format
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
        Tokenize sequences for OntoProtein model.

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
        pooling: Literal["mean", "cls", "max", "per_residue"] = "cls",  # CLS recommended for OntoProtein
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Extract embeddings from OntoProtein model.

        NOTE: For OntoProtein, "cls" pooling is recommended as the GO knowledge
        is primarily encoded in the [CLS] token representation.

        Args:
            sequences: List of amino acid sequences (can be contiguous or space-separated)
            pooling:
                - "cls": Use [CLS] token embedding (RECOMMENDED for OntoProtein)
                - "mean": Average over all residue positions
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
            # [CLS] token is at position 0
            # For OntoProtein, this is recommended as GO knowledge is in [CLS]
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
        # Use CLS pooling for OntoProtein (GO knowledge is in [CLS] token)
        embeddings = self.get_embeddings(sequences, pooling="cls")

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
