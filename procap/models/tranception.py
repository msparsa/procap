"""
Tranception Model Adapter for ProCap Benchmark.

Tranception is a protein language model specialized for mutation effect prediction.
Published at ICML 2022, it uses an autoregressive transformer with inference-time retrieval.

Key characteristics:
- Specialized for variant effect prediction (fitness landscapes)
- Uses autoregressive scoring of mutations
- Handles both single mutations and multiple mutations
- Superior performance on shallow alignments

Usage:
    model = TranceptionAdapter()
    model.load()
    # Score mutations
    scores = model.score_mutations(
        wild_type="MLKFV...",
        mutations=["M1A", "L2V", "K3E"]
    )
    # Or get embeddings
    embeddings = model.get_embeddings(["MLKFV"], pooling="mean")
"""

from typing import List, Literal, Optional, Dict, Any
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from procap.core.base_model import BaseProteinModel, ModelOutput, TokenizerOutput


class TranceptionAdapter(BaseProteinModel):
    """
    Adapter for Tranception mutation effect prediction model.

    Tranception scores mutations by computing log-likelihood ratios
    between wild-type and mutant sequences.
    """

    MODEL_VARIANTS = {
        "tranception_small": "OATML-Markslab/Tranception_Small",
        "tranception_medium": "OATML-Markslab/Tranception_Medium",
        "tranception_large": "OATML-Markslab/Tranception_Large",
    }

    def __init__(
        self,
        variant: str = "tranception_medium",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 8,
        max_sequence_length: int = 1024,
        **kwargs,
    ):
        """
        Initialize Tranception adapter.

        Args:
            variant: Model variant (default "tranception_medium")
            device: Device to use (None for auto-detect)
            dtype: Data type for model weights
            max_batch_size: Maximum batch size
            max_sequence_length: Maximum sequence length
        """
        name = f"Tranception-{variant}"
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
        """Load Tranception model and tokenizer."""
        print(f"Loading {self.name} from {self.repo_id}...")

        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.repo_id,
                trust_remote_code=True,
            )

            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.repo_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)

            self._model.eval()

            # Set model properties
            if hasattr(self._model.config, "n_embd"):
                self.hidden_size = self._model.config.n_embd
            elif hasattr(self._model.config, "hidden_size"):
                self.hidden_size = self._model.config.hidden_size
            else:
                self.hidden_size = 1024  # Default

            if hasattr(self._model.config, "n_layer"):
                self.num_layers = self._model.config.n_layer
            elif hasattr(self._model.config, "num_hidden_layers"):
                self.num_layers = self._model.config.num_hidden_layers
            else:
                self.num_layers = 24  # Default

            self._is_loaded = True
            print(f"Loaded {self.name}: hidden_size={self.hidden_size}, layers={self.num_layers}")

        except Exception as e:
            print(f"Warning: Could not load Tranception from HuggingFace: {e}")
            print("Tranception may require manual installation from the official repo.")
            print("See: https://github.com/OATML-Markslab/Tranception")
            raise

    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess sequence for Tranception."""
        # Remove spaces, convert to uppercase
        sequence = sequence.upper().replace(" ", "")

        # Truncate if needed
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[: self.max_sequence_length]

        return sequence

    def tokenize(self, sequences: List[str]) -> TokenizerOutput:
        """Tokenize sequences for Tranception model."""
        # Preprocess sequences
        processed_seqs = [self._preprocess_sequence(seq) for seq in sequences]

        # Tokenize
        encoded = self._tokenizer(
            processed_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

        # Calculate sequence lengths
        sequence_lengths = [len(seq) for seq in processed_seqs]

        return TokenizerOutput(
            input_ids=encoded["input_ids"].to(self.device),
            attention_mask=encoded["attention_mask"].to(self.device),
            sequence_lengths=sequence_lengths,
        )

    def get_embeddings(
        self,
        sequences: List[str],
        pooling: Literal["mean", "last", "max", "per_residue"] = "mean",
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Extract embeddings from Tranception model.

        Args:
            sequences: List of amino acid sequences
            pooling:
                - "mean": Average over all positions
                - "last": Use last token embedding
                - "max": Max pooling
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
                output_hidden_states=True,
            )

        # Get hidden states from specified layer
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[layer]
        else:
            raise RuntimeError("Model did not return hidden states")

        if pooling == "per_residue":
            return hidden_states

        elif pooling == "last":
            seq_lengths = tokens.attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            last_hidden = torch.stack([
                hidden_states[i, seq_lengths[i].item(), :]
                for i in range(batch_size)
            ])
            return last_hidden

        elif pooling == "mean":
            mask = tokens.attention_mask.unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            return summed / counts

        elif pooling == "max":
            mask = tokens.attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states_masked = hidden_states.clone()
            hidden_states_masked[mask == 0] = float("-inf")
            return hidden_states_masked.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

    def _parse_mutation(self, mutation: str) -> tuple:
        """
        Parse mutation string into (position, wild_type_aa, mutant_aa).

        Examples:
            "M1A" -> (1, "M", "A")
            "L123V" -> (123, "L", "V")
        """
        match = re.match(r"([A-Z])(\d+)([A-Z])", mutation.upper())
        if not match:
            raise ValueError(f"Invalid mutation format: {mutation}. Expected format like 'M1A' or 'L123V'")

        wild_aa = match.group(1)
        position = int(match.group(2))
        mut_aa = match.group(3)

        return position, wild_aa, mut_aa

    def _apply_mutation(self, sequence: str, mutation: str) -> str:
        """Apply a mutation to a sequence."""
        position, wild_aa, mut_aa = self._parse_mutation(mutation)

        # Convert to 0-indexed
        idx = position - 1

        if idx < 0 or idx >= len(sequence):
            raise ValueError(f"Mutation position {position} out of range for sequence length {len(sequence)}")

        if sequence[idx] != wild_aa:
            raise ValueError(f"Wild-type amino acid mismatch: expected {wild_aa} at position {position}, found {sequence[idx]}")

        return sequence[:idx] + mut_aa + sequence[idx + 1:]

    def _compute_log_likelihood(self, sequence: str) -> float:
        """Compute log-likelihood of a sequence."""
        tokens = self.tokenize([sequence])

        with torch.inference_mode():
            outputs = self._model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )

        # Get log probabilities
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Sum log probabilities of actual tokens (shifted by 1 for autoregressive)
        input_ids = tokens.input_ids[0, 1:]  # Exclude first token
        token_log_probs = log_probs[0, :-1]  # Exclude last position

        # Gather log probs for actual tokens
        selected_log_probs = token_log_probs.gather(1, input_ids.unsqueeze(-1)).squeeze(-1)

        # Apply attention mask
        mask = tokens.attention_mask[0, 1:].float()
        total_log_prob = (selected_log_probs * mask).sum().item()

        return total_log_prob

    def score_mutations(
        self,
        wild_type: str,
        mutations: List[str],
    ) -> List[float]:
        """
        Score mutations by computing log-likelihood ratio.

        Args:
            wild_type: Wild-type protein sequence
            mutations: List of mutations in format "X123Y" (e.g., "M1A", "L45V")

        Returns:
            List of mutation effect scores (log-likelihood ratios)
        """
        if not self._is_loaded:
            raise RuntimeError(f"Model {self.name} not loaded. Call load() first.")

        wild_type = self._preprocess_sequence(wild_type)

        # Compute wild-type log-likelihood
        wt_ll = self._compute_log_likelihood(wild_type)

        scores = []
        for mutation in mutations:
            try:
                # Apply mutation
                mutant = self._apply_mutation(wild_type, mutation)

                # Compute mutant log-likelihood
                mut_ll = self._compute_log_likelihood(mutant)

                # Score is log-likelihood ratio (mutant - wild_type)
                score = mut_ll - wt_ll
                scores.append(score)
            except Exception as e:
                print(f"Warning: Could not score mutation {mutation}: {e}")
                scores.append(0.0)

        return scores

    def _predict_single_batch(
        self,
        sequences: List[str],
        return_embeddings: bool = False,
    ) -> ModelOutput:
        """Run prediction on a single batch."""
        output = ModelOutput(metadata={"model": self.name, "variant": self.variant})

        if return_embeddings:
            embeddings = self.get_embeddings(sequences, pooling="mean")
            output.embeddings = embeddings

        # Apply head if attached
        if self._head is not None:
            embeddings = output.embeddings
            if embeddings is None:
                embeddings = self.get_embeddings(sequences, pooling="mean")
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
