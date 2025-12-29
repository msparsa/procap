"""
ProCap: Protein Capability Benchmark

A modular benchmark system for evaluating Protein Language Models
across Bloom's Taxonomy levels.
"""

__version__ = "2.0.0"
__author__ = "Mohammad Parsa"

from procap.core.base_model import BaseProteinModel, ModelOutput, TokenizerOutput
from procap.models.registry import ModelRegistry, get_model
from procap.tasks.registry import TaskRegistry, get_task

__all__ = [
    "BaseProteinModel",
    "ModelOutput",
    "TokenizerOutput",
    "ModelRegistry",
    "TaskRegistry",
    "get_model",
    "get_task",
]
