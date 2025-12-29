"""Protein Language Model adapters."""

from procap.models.esm import ESMAdapter
from procap.models.protbert import ProtBERTAdapter
from procap.models.ontoprotein import OntoProteinAdapter
from procap.models.prott5 import ProtT5Adapter
from procap.models.ankh import AnkhAdapter
from procap.models.protalbert import ProtAlbertAdapter
from procap.models.progen2 import ProGen2Adapter
from procap.models.protgpt2 import ProtGPT2Adapter
from procap.models.zymctrl import ZymCTRLAdapter
from procap.models.registry import ModelRegistry, get_model
from procap.models.heads import (
    MultiLabelClassificationHead,
    BinaryClassificationHead,
    RegressionHead,
)

__all__ = [
    # Model Adapters
    "ESMAdapter",
    "ProtBERTAdapter",
    "OntoProteinAdapter",
    "ProtT5Adapter",
    "AnkhAdapter",
    "ProtAlbertAdapter",
    "ProGen2Adapter",
    "ProtGPT2Adapter",
    "ZymCTRLAdapter",
    # Registry
    "ModelRegistry",
    "get_model",
    # Heads
    "MultiLabelClassificationHead",
    "BinaryClassificationHead",
    "RegressionHead",
]
