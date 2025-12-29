"""Task runners for ProCap benchmark."""

from procap.runners.base import BaseRunner
from procap.runners.classification import (
    MultiLabelClassificationRunner,
    MulticlassClassificationRunner,
    BinaryClassificationRunner,
)
from procap.runners.regression import RegressionRunner
from procap.runners.generation import TextGenerationRunner, SequenceGenerationRunner
from procap.runners.token_classification import TokenClassificationRunner
from procap.runners.ppi import PPIClassificationRunner, HomologyDetectionRunner

__all__ = [
    "BaseRunner",
    "MultiLabelClassificationRunner",
    "MulticlassClassificationRunner",
    "BinaryClassificationRunner",
    "RegressionRunner",
    "TextGenerationRunner",
    "SequenceGenerationRunner",
    "TokenClassificationRunner",
    "PPIClassificationRunner",
    "HomologyDetectionRunner",
]
