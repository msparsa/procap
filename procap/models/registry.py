"""
Model registry and factory for ProCap benchmark.

Provides a unified interface for loading and managing protein language models.

Usage:
    # Get a model by name
    model = get_model("esm2_t33_650M")
    model.load()

    # Or use the registry
    registry = ModelRegistry()
    registry.register("my_model", MyModelAdapter, {"param": "value"})
    model = registry.get("my_model")
"""

from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
import yaml

from procap.core.base_model import BaseProteinModel
from procap.models.esm import ESMAdapter
from procap.models.protbert import ProtBERTAdapter
from procap.models.ontoprotein import OntoProteinAdapter
from procap.models.prott5 import ProtT5Adapter
from procap.models.ankh import AnkhAdapter
from procap.models.protalbert import ProtAlbertAdapter
from procap.models.progen2 import ProGen2Adapter
from procap.models.zymctrl import ZymCTRLAdapter
from procap.models.saprot import SaProtAdapter
from procap.models.protgpt2 import ProtGPT2Adapter
from procap.models.tranception import TranceptionAdapter


# Default model configurations
DEFAULT_MODELS: Dict[str, Dict[str, Any]] = {
    # ESM-2 variants
    "esm2_t6_8M": {
        "class": ESMAdapter,
        "params": {"variant": "esm2_t6_8M", "max_batch_size": 64},
    },
    "esm2_t12_35M": {
        "class": ESMAdapter,
        "params": {"variant": "esm2_t12_35M", "max_batch_size": 32},
    },
    "esm2_t30_150M": {
        "class": ESMAdapter,
        "params": {"variant": "esm2_t30_150M", "max_batch_size": 16},
    },
    "esm2_t33_650M": {
        "class": ESMAdapter,
        "params": {"variant": "esm2_t33_650M", "max_batch_size": 8},
    },
    "esm2_t36_3B": {
        "class": ESMAdapter,
        "params": {"variant": "esm2_t36_3B", "max_batch_size": 4},
    },
    "esm2_t48_15B": {
        "class": ESMAdapter,
        "params": {"variant": "esm2_t48_15B", "max_batch_size": 1},
    },
    # ESM-1b
    "esm1b_t33_650M": {
        "class": ESMAdapter,
        "params": {"variant": "esm1b_t33_650M", "max_batch_size": 8},
    },
    # ESM-1v variants (for variant effect prediction)
    "esm1v_t33_650M_1": {
        "class": ESMAdapter,
        "params": {"variant": "esm1v_t33_650M_1", "max_batch_size": 8},
    },
    "esm1v_t33_650M_2": {
        "class": ESMAdapter,
        "params": {"variant": "esm1v_t33_650M_2", "max_batch_size": 8},
    },
    "esm1v_t33_650M_3": {
        "class": ESMAdapter,
        "params": {"variant": "esm1v_t33_650M_3", "max_batch_size": 8},
    },
    "esm1v_t33_650M_4": {
        "class": ESMAdapter,
        "params": {"variant": "esm1v_t33_650M_4", "max_batch_size": 8},
    },
    "esm1v_t33_650M_5": {
        "class": ESMAdapter,
        "params": {"variant": "esm1v_t33_650M_5", "max_batch_size": 8},
    },
    # ProtBERT variants
    "protbert": {
        "class": ProtBERTAdapter,
        "params": {"variant": "prot_bert", "max_batch_size": 32},
    },
    "protbert_bfd": {
        "class": ProtBERTAdapter,
        "params": {"variant": "prot_bert_bfd", "max_batch_size": 32},
    },
    # OntoProtein
    "ontoprotein": {
        "class": OntoProteinAdapter,
        "params": {"variant": "ontoprotein", "max_batch_size": 32},
    },
    # ProtT5 variants
    "prott5_xl_bfd": {
        "class": ProtT5Adapter,
        "params": {"variant": "prott5_xl_bfd", "max_batch_size": 4},
    },
    "prott5_xl_uniref50": {
        "class": ProtT5Adapter,
        "params": {"variant": "prott5_xl_uniref50", "max_batch_size": 4},
    },
    "prott5_xl_half": {
        "class": ProtT5Adapter,
        "params": {"variant": "prott5_xl_half", "max_batch_size": 8},
    },
    # Ankh variants
    "ankh_base": {
        "class": AnkhAdapter,
        "params": {"variant": "ankh_base", "max_batch_size": 16},
    },
    "ankh_large": {
        "class": AnkhAdapter,
        "params": {"variant": "ankh_large", "max_batch_size": 8},
    },
    # ProtAlbert
    "prot_albert": {
        "class": ProtAlbertAdapter,
        "params": {"variant": "prot_albert", "max_batch_size": 64},
    },
    # ProGen2 variants (generative)
    "progen2_small": {
        "class": ProGen2Adapter,
        "params": {"variant": "progen2_small", "max_batch_size": 16},
    },
    "progen2_medium": {
        "class": ProGen2Adapter,
        "params": {"variant": "progen2_medium", "max_batch_size": 8},
    },
    "progen2_base": {
        "class": ProGen2Adapter,
        "params": {"variant": "progen2_base", "max_batch_size": 8},
    },
    "progen2_large": {
        "class": ProGen2Adapter,
        "params": {"variant": "progen2_large", "max_batch_size": 4},
    },
    # ZymCTRL (enzyme generation)
    "zymctrl": {
        "class": ZymCTRLAdapter,
        "params": {"variant": "zymctrl", "max_batch_size": 8},
    },
    # SaProt (structure-aware, ICLR 2024)
    "saprot_650m_af2": {
        "class": SaProtAdapter,
        "params": {"variant": "saprot_650m_af2", "max_batch_size": 8},
    },
    "saprot_650m_pdb": {
        "class": SaProtAdapter,
        "params": {"variant": "saprot_650m_pdb", "max_batch_size": 8},
    },
    # ProtGPT2 (generative, Nature Comm 2022)
    "protgpt2": {
        "class": ProtGPT2Adapter,
        "params": {"variant": "protgpt2", "max_batch_size": 8},
    },
    # Tranception (mutation effect prediction, ICML 2022)
    "tranception_small": {
        "class": TranceptionAdapter,
        "params": {"variant": "tranception_small", "max_batch_size": 8},
    },
    "tranception_medium": {
        "class": TranceptionAdapter,
        "params": {"variant": "tranception_medium", "max_batch_size": 4},
    },
    "tranception_large": {
        "class": TranceptionAdapter,
        "params": {"variant": "tranception_large", "max_batch_size": 2},
    },
}

# Alias mappings for common names
MODEL_ALIASES: Dict[str, str] = {
    # ESM aliases
    "esm2": "esm2_t33_650M",
    "esm-2": "esm2_t33_650M",
    "esm2-650m": "esm2_t33_650M",
    "esm2-3b": "esm2_t36_3B",
    "esm1b": "esm1b_t33_650M",
    "esm1v": "esm1v_t33_650M_1",
    # ProtBERT aliases
    "prot_bert": "protbert",
    "ProtBERT": "protbert",
    "prot_bert_bfd": "protbert_bfd",
    # OntoProtein aliases
    "OntoProtein": "ontoprotein",
    # ProtT5 aliases
    "prott5": "prott5_xl_bfd",
    "prot_t5": "prott5_xl_bfd",
    "ProtT5": "prott5_xl_bfd",
    # Ankh aliases
    "ankh": "ankh_base",
    "Ankh": "ankh_base",
    # ProtAlbert aliases
    "protalbert": "prot_albert",
    "ProtAlbert": "prot_albert",
    # ProGen2 aliases
    "progen2": "progen2_small",
    "ProGen2": "progen2_small",
    # ZymCTRL aliases
    "ZymCTRL": "zymctrl",
    # SaProt aliases
    "saprot": "saprot_650m_af2",
    "SaProt": "saprot_650m_af2",
    # ProtGPT2 aliases
    "ProtGPT2": "protgpt2",
    "prot_gpt2": "protgpt2",
    # Tranception aliases
    "tranception": "tranception_medium",
    "Tranception": "tranception_medium",
}


class ModelRegistry:
    """
    Registry for protein language models.

    Manages model configurations, instantiation, and caching.

    Usage:
        registry = ModelRegistry()

        # Get available models
        print(registry.list_models())

        # Get a model instance
        model = registry.get("esm2_t33_650M")
        model.load()

        # Register custom model
        registry.register("my_model", MyAdapter, {"variant": "custom"})
    """

    def __init__(self, load_defaults: bool = True):
        """
        Initialize model registry.

        Args:
            load_defaults: Whether to load default model configurations
        """
        self._models: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[str, BaseProteinModel] = {}

        if load_defaults:
            for name, config in DEFAULT_MODELS.items():
                self._models[name] = config

    def register(
        self,
        name: str,
        model_class: Type[BaseProteinModel],
        params: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a model configuration.

        Args:
            name: Model identifier
            model_class: Model adapter class
            params: Constructor parameters
            overwrite: Whether to overwrite existing registration
        """
        if name in self._models and not overwrite:
            raise ValueError(f"Model {name!r} already registered. Use overwrite=True to replace.")

        self._models[name] = {
            "class": model_class,
            "params": params or {},
        }

    def get(
        self,
        name: str,
        use_cache: bool = True,
        **override_params,
    ) -> BaseProteinModel:
        """
        Get a model instance by name.

        Args:
            name: Model identifier (or alias)
            use_cache: Whether to return cached instance
            **override_params: Parameters to override

        Returns:
            Model instance (not yet loaded - call .load())
        """
        # Resolve alias
        resolved_name = MODEL_ALIASES.get(name, name)

        # Check cache
        if use_cache and resolved_name in self._cache:
            return self._cache[resolved_name]

        # Get configuration
        if resolved_name not in self._models:
            available = ", ".join(self.list_models())
            raise ValueError(
                f"Unknown model {name!r}. Available models: {available}"
            )

        config = self._models[resolved_name]
        model_class = config["class"]
        params = {**config["params"], **override_params}

        # Instantiate model
        model = model_class(**params)

        # Cache if requested
        if use_cache:
            self._cache[resolved_name] = model

        return model

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def list_aliases(self) -> Dict[str, str]:
        """List all model aliases."""
        return MODEL_ALIASES.copy()

    def get_config(self, name: str) -> Dict[str, Any]:
        """Get configuration for a model."""
        resolved_name = MODEL_ALIASES.get(name, name)
        if resolved_name not in self._models:
            raise ValueError(f"Unknown model {name!r}")
        return self._models[resolved_name].copy()

    def clear_cache(self) -> None:
        """Clear cached model instances."""
        self._cache.clear()

    def load_from_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Load model configurations from a YAML file.

        YAML format:
            models:
              my_esm:
                type: esm
                variant: esm2_t33_650M
                max_batch_size: 8

              my_protbert:
                type: protbert
                variant: prot_bert
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        type_to_class = {
            "esm": ESMAdapter,
            "protbert": ProtBERTAdapter,
            "ontoprotein": OntoProteinAdapter,
            "prott5": ProtT5Adapter,
            "ankh": AnkhAdapter,
            "protalbert": ProtAlbertAdapter,
            "progen2": ProGen2Adapter,
            "zymctrl": ZymCTRLAdapter,
            "saprot": SaProtAdapter,
            "protgpt2": ProtGPT2Adapter,
            "tranception": TranceptionAdapter,
        }

        for name, model_config in config.get("models", {}).items():
            model_type = model_config.pop("type", "esm")
            model_class = type_to_class.get(model_type)

            if model_class is None:
                raise ValueError(f"Unknown model type {model_type!r}")

            self.register(name, model_class, model_config, overwrite=True)


# Global registry instance
_global_registry = ModelRegistry()


def get_model(name: str, **kwargs) -> BaseProteinModel:
    """
    Get a model instance from the global registry.

    Args:
        name: Model identifier (or alias)
        **kwargs: Override parameters

    Returns:
        Model instance (call .load() to load weights)

    Example:
        model = get_model("esm2_t33_650M")
        model.load()
        embeddings = model.get_embeddings(["MLKFV"])
    """
    return _global_registry.get(name, **kwargs)


def list_models() -> List[str]:
    """List all available models."""
    return _global_registry.list_models()


def register_model(
    name: str,
    model_class: Type[BaseProteinModel],
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Register a model in the global registry."""
    _global_registry.register(name, model_class, params)
