"""
ProGen2 Model Adapter for ProCap Benchmark.

ProGen2 is an autoregressive protein language model for protein sequence generation.
It is based on the GPT-2 architecture and trained on protein sequences.

Key characteristics:
- Autoregressive (decoder-only) architecture
- Can generate protein sequences
- Multiple sizes from 151M to 6.4B parameters
- Uses custom tokenizer

Usage:
    model = ProGen2Adapter(variant="progen2_small")
    model.load()
    # Generate new sequences
    generated = model.generate(["MKTV"], max_new_tokens=50)
    # Or get embeddings from internal representations
    embeddings = model.get_embeddings(["MLKFV", "ACDEFG"], pooling="mean")
"""

from typing import List, Literal, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from procap.core.base_model import BaseProteinModel, ModelOutput, TokenizerOutput


class ProGen2Adapter(BaseProteinModel):
    """
    Adapter for ProGen2 generative protein language models.

    ProGen2 is a decoder-only model optimized for protein sequence generation.
    It can both generate new sequences and provide embeddings.
    """

    MODEL_VARIANTS = {
        # ProGen2 models (various sizes)
        "progen2_small": "hugohrban/progen2-small",  # 151M params
        "progen2_medium": "hugohrban/progen2-medium",  # 764M params
        "progen2_base": "hugohrban/progen2-base",  # 764M params
        "progen2_large": "hugohrban/progen2-large",  # 2.7B params
        "progen2_xlarge": "hugohrban/progen2-xlarge",  # 6.4B params
    }

    def __init__(
        self,
        variant: str = "progen2_small",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 8,
        max_sequence_length: int = 1024,
        **kwargs,
    ):
        """
        Initialize ProGen2 adapter.

        Args:
            variant: Model variant (default "progen2_small")
            device: Device to use (None for auto-detect)
            dtype: Data type for model weights
            max_batch_size: Maximum batch size
            max_sequence_length: Maximum sequence length for generation
        """
        name = f"ProGen2-{variant}"
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
        """Load ProGen2 model and tokenizer from HuggingFace."""
        print(f"Loading {self.name} from {self.repo_id}...")

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
        self.hidden_size = self._model.config.n_embd
        self.num_layers = self._model.config.n_layer
        self._is_loaded = True

        print(f"Loaded {self.name}: hidden_size={self.hidden_size}, layers={self.num_layers}")

    def _preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess sequence for ProGen2.

        ProGen2 uses contiguous sequences (no spaces).
        """
        # Remove spaces, convert to uppercase
        sequence = sequence.upper().replace(" ", "")

        # Truncate if needed
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[: self.max_sequence_length]

        return sequence

    def tokenize(self, sequences: List[str]) -> TokenizerOutput:
        """Tokenize sequences for ProGen2 model."""
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
        Extract embeddings from ProGen2 model.

        For decoder-only models, we use the hidden states from forward pass.

        Args:
            sequences: List of amino acid sequences
            pooling:
                - "mean": Average over all positions
                - "last": Use last token embedding (common for decoders)
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
            # Get last non-padded token for each sequence
            # Find the last valid position for each sequence
            seq_lengths = tokens.attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            batch_size = hidden_states.shape[0]
            last_hidden = torch.stack([
                hidden_states[i, seq_lengths[i].item(), :]
                for i in range(batch_size)
            ])
            return last_hidden

        elif pooling == "mean":
            # Mean pooling over all positions (excluding padding)
            mask = tokens.attention_mask.unsqueeze(-1).float()
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

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> List[str]:
        """
        Generate protein sequences.

        Args:
            prompts: List of sequence prompts to continue
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)
            num_return_sequences: Number of sequences to generate per prompt

        Returns:
            List of generated sequences
        """
        if not self._is_loaded:
            raise RuntimeError(f"Model {self.name} not loaded. Call load() first.")

        # Preprocess prompts
        processed_prompts = [self._preprocess_sequence(p) for p in prompts]

        # Tokenize
        encoded = self._tokenizer(
            processed_prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self._tokenizer.pad_token_id,
                **kwargs,
            )

        # Decode generated sequences
        generated_texts = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts

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

        # For generative models, generate continuations
        generated = self.generate(sequences, max_new_tokens=50)
        output.generated_text = generated

        # Apply head if attached (for classification/regression tasks)
        if self._head is not None and return_embeddings:
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
