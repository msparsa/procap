"""Data loading utilities for ProCap benchmark."""

from procap.data.loader import DataLoader, load_dataset
from procap.data.swiss_prot import SwissProtLoader

__all__ = [
    "DataLoader",
    "load_dataset",
    "SwissProtLoader",
]
