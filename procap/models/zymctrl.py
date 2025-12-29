"""
ZymCTRL Model Adapter for ProCap Benchmark.

ZymCTRL is an autoregressive protein language model specialized for enzyme sequences.
It can generate enzyme sequences conditioned on EC (Enzyme Commission) numbers.

Key characteristics:
- GPT-2 architecture trained on 37M enzyme sequences
- Can generate sequences conditioned on EC class
- Useful for enzyme engineering tasks

Usage:
    model = ZymCTRLAdapter()
    model.load()
    # Generate enzyme sequences (optionally conditioned on EC)
    generated = model.generate(["1.1.1.1"], max_new_tokens=100)  # EC-conditioned
    # Or get embeddings
    embeddings = model.get_embeddings(["MLKFV"], pooling="mean")
"""

from typing import List, Literal, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from procap.core.base_model import BaseProteinModel, ModelOutput, TokenizerOutput


class ZymCTRLAdapter(BaseProteinModel):
    """
    Adapter for ZymCTRL enzyme sequence generation model.

    ZymCTRL can generate enzyme sequences optionally conditioned on EC numbers.
    """

    MODEL_VARIANTS = {
        "zymctrl": "AI4PD/ZymCTRL",
    }

    def __init__(
        self,
        variant: str = "zymctrl",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 8,
        max_sequence_length: int = 1024,
        **kwargs,
    ):
        """
        Initialize ZymCTRL adapter.

        Args:
            variant: Model variant (default "zymctrl")
            device: Device to use (None for auto-detect)
            dtype: Data type for model weights
            max_batch_size: Maximum batch size
            max_sequence_length: Maximum sequence length
        """
        name = "ZymCTRL"
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
        """Load ZymCTRL model and tokenizer from HuggingFace."""
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

        # Set model properties - ZymCTRL uses GPT-2 config
        self.hidden_size = self._model.config.n_embd
        self.num_layers = self._model.config.n_layer
        self._is_loaded = True

        print(f"Loaded {self.name}: hidden_size={self.hidden_size}, layers={self.num_layers}")

    def _preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess sequence for ZymCTRL.

        Sequences can be:
        - Plain protein sequences: "MKTV..."
        - EC-conditioned: "1.1.1.1<sep>MKTV..."
        """
        # Keep as-is if it looks like EC-conditioned input
        if "<sep>" in sequence or sequence[0].isdigit():
            return sequence

        # Otherwise, treat as plain sequence
        return sequence.upper().replace(" ", "")

    def tokenize(self, sequences: List[str]) -> TokenizerOutput:
        """Tokenize sequences for ZymCTRL model."""
        processed_seqs = [self._preprocess_sequence(seq) for seq in sequences]

        encoded = self._tokenizer(
            processed_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

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
        Extract embeddings from ZymCTRL model.

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

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 9,
        top_p: float = 1.0,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Generate enzyme sequences.

        Args:
            prompts: List of prompts. Can be:
                - EC numbers: "1.1.1.1" (will be formatted for generation)
                - Partial sequences: "MKTV"
                - EC + partial: "1.1.1.1<sep>MKTV"
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (ZymCTRL paper uses k=9)
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling

        Returns:
            List of generated sequences
        """
        if not self._is_loaded:
            raise RuntimeError(f"Model {self.name} not loaded. Call load() first.")

        # Format prompts for ZymCTRL
        formatted_prompts = []
        for prompt in prompts:
            # If it looks like an EC number, format it properly
            if prompt and prompt[0].isdigit() and "." in prompt and "<sep>" not in prompt:
                formatted_prompts.append(f"{prompt}<sep><start>")
            elif "<sep>" in prompt and "<start>" not in prompt:
                formatted_prompts.append(f"{prompt}<start>")
            else:
                formatted_prompts.append(prompt)

        encoded = self._tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

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
                pad_token_id=self._tokenizer.pad_token_id,
                **kwargs,
            )

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

        # Generate enzyme sequences
        generated = self.generate(sequences, max_new_tokens=50)
        output.generated_text = generated

        if self._head is not None and return_embeddings:
            embeddings = output.embeddings
            if embeddings is None:
                embeddings = self.get_embeddings(sequences, pooling="mean")
            head_output = self._head(embeddings)
            if isinstance(head_output, tuple):
                output.logits = head_output[0]
            else:
                output.logits = head_output

        return output

    @classmethod
    def list_variants(cls) -> List[str]:
        """List available model variants."""
        return list(cls.MODEL_VARIANTS.keys())
