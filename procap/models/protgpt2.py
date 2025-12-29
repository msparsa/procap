"""
ProtGPT2 Model Adapter for ProCap Benchmark.

ProtGPT2 is a generative protein language model from Nature Communications 2022.
It uses GPT-2 architecture for de novo protein sequence generation.

Key characteristics:
- Autoregressive (decoder-only) GPT-2 architecture
- Can generate protein sequences from scratch or continue prompts
- 738M parameters
- Uses space-separated amino acid tokens

Usage:
    model = ProtGPT2Adapter()
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


class ProtGPT2Adapter(BaseProteinModel):
    """
    Adapter for ProtGPT2 generative protein language model.

    ProtGPT2 is a decoder-only model for de novo protein sequence generation.
    It uses space-separated amino acid tokens.
    """

    MODEL_VARIANTS = {
        "protgpt2": "nferruz/ProtGPT2",
    }

    def __init__(
        self,
        variant: str = "protgpt2",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 8,
        max_sequence_length: int = 1024,
        **kwargs,
    ):
        """
        Initialize ProtGPT2 adapter.

        Args:
            variant: Model variant (default "protgpt2")
            device: Device to use (None for auto-detect)
            dtype: Data type for model weights
            max_batch_size: Maximum batch size
            max_sequence_length: Maximum sequence length for generation
        """
        name = f"ProtGPT2-{variant}"
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
        """Load ProtGPT2 model and tokenizer from HuggingFace."""
        print(f"Loading {self.name} from {self.repo_id}...")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.repo_id)

        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.repo_id,
            torch_dtype=self.dtype,
        ).to(self.device)

        self._model.eval()

        # Set model properties
        self.hidden_size = self._model.config.n_embd
        self.num_layers = self._model.config.n_layer
        self._is_loaded = True

        print(f"Loaded {self.name}: hidden_size={self.hidden_size}, layers={self.num_layers}")

    def _preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess sequence for ProtGPT2.

        ProtGPT2 uses space-separated amino acids.
        Converts "MLKFV" -> "M L K F V"
        """
        # Remove existing spaces and convert to uppercase
        sequence = sequence.upper().replace(" ", "")

        # Truncate if needed
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[: self.max_sequence_length]

        # Add spaces between amino acids
        return " ".join(list(sequence))

    def tokenize(self, sequences: List[str]) -> TokenizerOutput:
        """Tokenize sequences for ProtGPT2 model."""
        # Preprocess sequences (add spaces)
        processed_seqs = [self._preprocess_sequence(seq) for seq in sequences]

        # Tokenize
        encoded = self._tokenizer(
            processed_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length * 2,  # *2 for space tokens
            return_tensors="pt",
        )

        # Calculate original sequence lengths
        sequence_lengths = [len(seq.replace(" ", "")) for seq in processed_seqs]

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
        Extract embeddings from ProtGPT2 model.

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
        top_k: int = 950,
        top_p: float = 1.0,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> List[str]:
        """
        Generate protein sequences.

        Args:
            prompts: List of sequence prompts to continue (can be empty for de novo)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter (ProtGPT2 default is 950)
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (vs greedy decoding)
            num_return_sequences: Number of sequences to generate per prompt

        Returns:
            List of generated sequences
        """
        if not self._is_loaded:
            raise RuntimeError(f"Model {self.name} not loaded. Call load() first.")

        # Preprocess prompts (add spaces)
        processed_prompts = [self._preprocess_sequence(p) if p else "<|endoftext|>" for p in prompts]

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
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode generated sequences
        generated_texts = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Remove spaces from generated sequences
        generated_texts = [text.replace(" ", "") for text in generated_texts]

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
