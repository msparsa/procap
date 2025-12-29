"""
Tests for protein language model adapters.

These tests verify that each model adapter:
1. Tokenizes sequences correctly (with proper format)
2. Produces embeddings of the correct shape
3. Works with the model registry
"""

import pytest
import torch


class TestESMAdapter:
    """Tests for ESM-2 model adapter."""

    def test_preprocess_sequence_removes_spaces(self):
        """ESM uses contiguous sequences - spaces should be removed."""
        from procap.models.esm import ESMAdapter

        adapter = ESMAdapter(variant="esm2_t6_8M")
        seq_with_spaces = "M L K F V"
        processed = adapter._preprocess_sequence(seq_with_spaces)

        assert " " not in processed
        assert processed == "MLKFV"

    def test_preprocess_sequence_uppercase(self):
        """ESM should convert to uppercase."""
        from procap.models.esm import ESMAdapter

        adapter = ESMAdapter(variant="esm2_t6_8M")
        seq_lower = "mlkfv"
        processed = adapter._preprocess_sequence(seq_lower)

        assert processed == "MLKFV"

    def test_list_variants(self):
        """Test listing available variants."""
        from procap.models.esm import ESMAdapter

        variants = ESMAdapter.list_variants()
        assert "esm2_t6_8M" in variants
        assert "esm2_t33_650M" in variants

    @pytest.mark.slow
    def test_load_and_tokenize(self, sample_sequences):
        """Test loading model and tokenizing sequences."""
        from procap.models.esm import ESMAdapter

        adapter = ESMAdapter(variant="esm2_t6_8M")
        adapter.load()

        tokens = adapter.tokenize(sample_sequences)

        assert tokens.input_ids.shape[0] == len(sample_sequences)
        assert tokens.attention_mask.shape == tokens.input_ids.shape

    @pytest.mark.slow
    def test_get_embeddings_shape(self, sample_sequences):
        """Test embedding extraction produces correct shape."""
        from procap.models.esm import ESMAdapter

        adapter = ESMAdapter(variant="esm2_t6_8M")
        adapter.load()

        embeddings = adapter.get_embeddings(sample_sequences, pooling="mean")

        assert embeddings.shape[0] == len(sample_sequences)
        assert embeddings.shape[1] == adapter.hidden_size


class TestProtBERTAdapter:
    """Tests for ProtBERT model adapter."""

    def test_preprocess_adds_spaces(self):
        """ProtBERT requires space-separated amino acids."""
        from procap.models.protbert import ProtBERTAdapter

        # Static method test
        seq = "MLKFV"
        processed = ProtBERTAdapter.preprocess_sequence(seq)

        assert processed == "M L K F V"

    def test_preprocess_replaces_rare_amino_acids(self):
        """ProtBERT replaces U, Z, O, B with X."""
        from procap.models.protbert import ProtBERTAdapter

        seq_with_rare = "MLUFVZBKO"
        processed = ProtBERTAdapter.preprocess_sequence(seq_with_rare)

        # U, Z, B, O should be replaced with X
        assert "U" not in processed
        assert "Z" not in processed
        assert "B" not in processed
        assert "O" not in processed

    def test_list_variants(self):
        """Test listing available variants."""
        from procap.models.protbert import ProtBERTAdapter

        variants = ProtBERTAdapter.list_variants()
        assert "prot_bert" in variants

    @pytest.mark.slow
    def test_load_and_tokenize(self, sample_sequences):
        """Test loading model and tokenizing sequences."""
        from procap.models.protbert import ProtBERTAdapter

        adapter = ProtBERTAdapter(variant="prot_bert")
        adapter.load()

        tokens = adapter.tokenize(sample_sequences)

        assert tokens.input_ids.shape[0] == len(sample_sequences)

    @pytest.mark.slow
    def test_get_embeddings_shape(self, sample_sequences):
        """Test embedding extraction produces correct shape."""
        from procap.models.protbert import ProtBERTAdapter

        adapter = ProtBERTAdapter(variant="prot_bert")
        adapter.load()

        embeddings = adapter.get_embeddings(sample_sequences, pooling="mean")

        assert embeddings.shape[0] == len(sample_sequences)
        assert embeddings.shape[1] == adapter.hidden_size


class TestOntoProteinAdapter:
    """Tests for OntoProtein model adapter."""

    def test_preprocess_adds_spaces(self):
        """OntoProtein requires space-separated amino acids."""
        from procap.models.ontoprotein import OntoProteinAdapter

        seq = "MLKFV"
        processed = OntoProteinAdapter.preprocess_sequence(seq)

        assert processed == "M L K F V"

    @pytest.mark.slow
    def test_load_and_tokenize(self, sample_sequences):
        """Test loading model and tokenizing sequences."""
        from procap.models.ontoprotein import OntoProteinAdapter

        adapter = OntoProteinAdapter()
        adapter.load()

        tokens = adapter.tokenize(sample_sequences)

        assert tokens.input_ids.shape[0] == len(sample_sequences)

    @pytest.mark.slow
    def test_cls_pooling_for_go(self, sample_sequences):
        """OntoProtein uses CLS pooling for GO tasks."""
        from procap.models.ontoprotein import OntoProteinAdapter

        adapter = OntoProteinAdapter()
        adapter.load()

        # Default pooling should be CLS for OntoProtein
        embeddings = adapter.get_embeddings(sample_sequences, pooling="cls")

        assert embeddings.shape[0] == len(sample_sequences)
        assert embeddings.shape[1] == adapter.hidden_size


class TestProtT5Adapter:
    """Tests for ProtT5 model adapter."""

    def test_preprocess_adds_spaces(self):
        """ProtT5 requires space-separated amino acids."""
        from procap.models.prott5 import ProtT5Adapter

        seq = "MLKFV"
        processed = ProtT5Adapter.preprocess_sequence(seq)

        assert processed == "M L K F V"

    def test_preprocess_replaces_rare_amino_acids(self):
        """ProtT5 replaces U, Z, O, B with X."""
        from procap.models.prott5 import ProtT5Adapter

        seq_with_rare = "MLUFVZBKO"
        processed = ProtT5Adapter.preprocess_sequence(seq_with_rare)

        assert "U" not in processed
        assert "Z" not in processed
        assert "B" not in processed
        assert "O" not in processed

    def test_list_variants(self):
        """Test listing available variants."""
        from procap.models.prott5 import ProtT5Adapter

        variants = ProtT5Adapter.list_variants()
        assert "prott5_xl_bfd" in variants
        assert "prott5_xl_uniref50" in variants

    @pytest.mark.slow
    def test_load_and_tokenize(self, sample_sequences):
        """Test loading model and tokenizing sequences."""
        from procap.models.prott5 import ProtT5Adapter

        adapter = ProtT5Adapter(variant="prott5_xl_half")
        adapter.load()

        tokens = adapter.tokenize(sample_sequences)

        assert tokens.input_ids.shape[0] == len(sample_sequences)

    @pytest.mark.slow
    def test_get_embeddings_shape(self, sample_sequences):
        """Test embedding extraction produces correct shape."""
        from procap.models.prott5 import ProtT5Adapter

        adapter = ProtT5Adapter(variant="prott5_xl_half")
        adapter.load()

        embeddings = adapter.get_embeddings(sample_sequences, pooling="mean")

        assert embeddings.shape[0] == len(sample_sequences)
        assert embeddings.shape[1] == adapter.hidden_size


class TestAnkhAdapter:
    """Tests for Ankh model adapter."""

    def test_preprocess_adds_spaces(self):
        """Ankh requires space-separated amino acids."""
        from procap.models.ankh import AnkhAdapter

        seq = "MLKFV"
        processed = AnkhAdapter.preprocess_sequence(seq)

        assert processed == "M L K F V"

    def test_list_variants(self):
        """Test listing available variants."""
        from procap.models.ankh import AnkhAdapter

        variants = AnkhAdapter.list_variants()
        assert "ankh_base" in variants
        assert "ankh_large" in variants

    @pytest.mark.slow
    def test_load_and_tokenize(self, sample_sequences):
        """Test loading model and tokenizing sequences."""
        from procap.models.ankh import AnkhAdapter

        adapter = AnkhAdapter(variant="ankh_base")
        adapter.load()

        tokens = adapter.tokenize(sample_sequences)

        assert tokens.input_ids.shape[0] == len(sample_sequences)

    @pytest.mark.slow
    def test_get_embeddings_shape(self, sample_sequences):
        """Test embedding extraction produces correct shape."""
        from procap.models.ankh import AnkhAdapter

        adapter = AnkhAdapter(variant="ankh_base")
        adapter.load()

        embeddings = adapter.get_embeddings(sample_sequences, pooling="mean")

        assert embeddings.shape[0] == len(sample_sequences)
        assert embeddings.shape[1] == adapter.hidden_size


class TestProtAlbertAdapter:
    """Tests for ProtAlbert model adapter."""

    def test_preprocess_adds_spaces(self):
        """ProtAlbert requires space-separated amino acids."""
        from procap.models.protalbert import ProtAlbertAdapter

        seq = "MLKFV"
        processed = ProtAlbertAdapter.preprocess_sequence(seq)

        assert processed == "M L K F V"

    def test_list_variants(self):
        """Test listing available variants."""
        from procap.models.protalbert import ProtAlbertAdapter

        variants = ProtAlbertAdapter.list_variants()
        assert "prot_albert" in variants

    @pytest.mark.slow
    def test_load_and_tokenize(self, sample_sequences):
        """Test loading model and tokenizing sequences."""
        from procap.models.protalbert import ProtAlbertAdapter

        adapter = ProtAlbertAdapter()
        adapter.load()

        tokens = adapter.tokenize(sample_sequences)

        assert tokens.input_ids.shape[0] == len(sample_sequences)

    @pytest.mark.slow
    def test_get_embeddings_shape(self, sample_sequences):
        """Test embedding extraction produces correct shape."""
        from procap.models.protalbert import ProtAlbertAdapter

        adapter = ProtAlbertAdapter()
        adapter.load()

        embeddings = adapter.get_embeddings(sample_sequences, pooling="mean")

        assert embeddings.shape[0] == len(sample_sequences)
        assert embeddings.shape[1] == adapter.hidden_size


class TestProGen2Adapter:
    """Tests for ProGen2 generative model adapter."""

    def test_list_variants(self):
        """Test listing available variants."""
        from procap.models.progen2 import ProGen2Adapter

        variants = ProGen2Adapter.list_variants()
        assert "progen2_small" in variants
        assert "progen2_medium" in variants

    @pytest.mark.slow
    def test_load_and_tokenize(self, sample_sequences):
        """Test loading model and tokenizing sequences."""
        from procap.models.progen2 import ProGen2Adapter

        adapter = ProGen2Adapter(variant="progen2_small")
        adapter.load()

        tokens = adapter.tokenize(sample_sequences)

        assert tokens.input_ids.shape[0] == len(sample_sequences)

    @pytest.mark.slow
    def test_get_embeddings_shape(self, sample_sequences):
        """Test embedding extraction produces correct shape."""
        from procap.models.progen2 import ProGen2Adapter

        adapter = ProGen2Adapter(variant="progen2_small")
        adapter.load()

        embeddings = adapter.get_embeddings(sample_sequences, pooling="mean")

        assert embeddings.shape[0] == len(sample_sequences)
        assert embeddings.shape[1] == adapter.hidden_size

    @pytest.mark.slow
    def test_generate(self, sample_sequences):
        """Test sequence generation."""
        from procap.models.progen2 import ProGen2Adapter

        adapter = ProGen2Adapter(variant="progen2_small")
        adapter.load()

        # Generate from short prompts
        prompts = ["MKTV", "ACDE"]
        generated = adapter.generate(prompts, max_new_tokens=10)

        assert len(generated) == len(prompts)
        for seq in generated:
            assert len(seq) > 0


class TestZymCTRLAdapter:
    """Tests for ZymCTRL enzyme generation adapter."""

    def test_list_variants(self):
        """Test listing available variants."""
        from procap.models.zymctrl import ZymCTRLAdapter

        variants = ZymCTRLAdapter.list_variants()
        assert "zymctrl" in variants

    @pytest.mark.slow
    def test_load_and_tokenize(self, sample_sequences):
        """Test loading model and tokenizing sequences."""
        from procap.models.zymctrl import ZymCTRLAdapter

        adapter = ZymCTRLAdapter()
        adapter.load()

        tokens = adapter.tokenize(sample_sequences)

        assert tokens.input_ids.shape[0] == len(sample_sequences)

    @pytest.mark.slow
    def test_get_embeddings_shape(self, sample_sequences):
        """Test embedding extraction produces correct shape."""
        from procap.models.zymctrl import ZymCTRLAdapter

        adapter = ZymCTRLAdapter()
        adapter.load()

        embeddings = adapter.get_embeddings(sample_sequences, pooling="mean")

        assert embeddings.shape[0] == len(sample_sequences)
        assert embeddings.shape[1] == adapter.hidden_size


class TestModelRegistry:
    """Tests for model registry."""

    def test_list_models(self):
        """Test listing registered models."""
        from procap.models.registry import ModelRegistry

        registry = ModelRegistry()
        models = registry.list_models()

        assert "esm2_t33_650M" in models
        assert "protbert" in models
        assert "ontoprotein" in models
        # Check new models
        assert "prott5_xl_bfd" in models
        assert "ankh_base" in models
        assert "prot_albert" in models
        assert "progen2_small" in models
        assert "zymctrl" in models

    def test_get_model_by_name(self):
        """Test getting model by name."""
        from procap.models.registry import get_model

        model = get_model("esm2_t6_8M", use_cache=False)

        assert model is not None
        assert "ESM" in model.name

    def test_get_model_by_alias(self):
        """Test getting model by alias."""
        from procap.models.registry import get_model

        model = get_model("esm2", use_cache=False)

        assert model is not None

    def test_unknown_model_raises(self):
        """Test that unknown model raises ValueError."""
        from procap.models.registry import get_model

        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent_model")


class TestTaskHeads:
    """Tests for task-specific heads."""

    def test_multilabel_head_shape(self, mock_embeddings):
        """Test multi-label classification head output shape."""
        from procap.models.heads import MultiLabelClassificationHead

        head = MultiLabelClassificationHead(
            hidden_size=768,
            num_labels=100,
        )

        logits = head(mock_embeddings)

        assert logits.shape == (4, 100)

    def test_regression_head_shape(self, mock_embeddings):
        """Test regression head output shape."""
        from procap.models.heads import RegressionHead

        head = RegressionHead(hidden_size=768, num_outputs=1)

        predictions = head(mock_embeddings)

        assert predictions.shape == (4, 1)

    def test_binary_head_shape(self, mock_embeddings):
        """Test binary classification head output shape."""
        from procap.models.heads import BinaryClassificationHead

        head = BinaryClassificationHead(hidden_size=768)

        logits = head(mock_embeddings)

        assert logits.shape == (4,)
