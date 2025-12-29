"""
Pytest fixtures for ProCap benchmark tests.
"""

import pytest
import pandas as pd
import numpy as np
import torch


@pytest.fixture
def sample_sequences():
    """Sample protein sequences for testing."""
    return [
        "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGP",
        "MLSRVLNRAEWLVSRRQICLSSVR",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    ]


@pytest.fixture
def sample_go_terms():
    """Sample GO term annotations."""
    return [
        "GO:0005524;GO:0006468;GO:0004672",
        "GO:0003677;GO:0005634",
        "GO:0005833;GO:0019825;GO:0020037",
    ]


@pytest.fixture
def sample_dataframe(sample_sequences, sample_go_terms):
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "protein_id": ["P1", "P2", "P3"],
        "sequence": sample_sequences,
        "go_terms": sample_go_terms,
        "family": ["PF00001", "PF00002", "PF00003"],
        "species": ["Homo sapiens", "Mus musculus", "Homo sapiens"],
    })


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing heads."""
    return torch.randn(4, 768)


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
