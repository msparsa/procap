"""
General data loading utilities for ProCap benchmark.

Supports multiple file formats: CSV, Parquet, TSV, JSON.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class DataLoader:
    """
    Unified data loader supporting multiple formats.

    Usage:
        df = DataLoader.load("data/my_data.csv")
        df = DataLoader.load("data/my_data.parquet", columns=["sequence", "go_terms"])
    """

    SUPPORTED_FORMATS = {".csv", ".parquet", ".tsv", ".json"}

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from file, auto-detecting format.

        Args:
            path: Path to data file
            columns: Columns to load (None for all)
            nrows: Maximum number of rows to load
            **kwargs: Additional arguments for pandas reader

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is unsupported
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path, nrows=nrows, **kwargs)
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=nrows, **kwargs)
        elif suffix == ".parquet":
            df = pd.read_parquet(path, columns=columns, **kwargs)
            if nrows is not None:
                df = df.head(nrows)
        elif suffix == ".json":
            df = pd.read_json(path, nrows=nrows, **kwargs)
        else:
            raise ValueError(
                f"Unsupported format: {suffix}. Supported: {cls.SUPPORTED_FORMATS}"
            )

        # Select columns if specified (for non-parquet)
        if columns is not None and suffix != ".parquet":
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            df = df[columns]

        return df

    @classmethod
    def validate_schema(
        cls,
        df: pd.DataFrame,
        required_columns: List[str],
        optional_columns: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate DataFrame has required columns.

        Args:
            df: DataFrame to validate
            required_columns: Columns that must exist
            optional_columns: Columns that may exist

        Returns:
            True if validation passes

        Raises:
            ValueError: If required columns are missing
        """
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

    @classmethod
    def get_info(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary information about a DataFrame.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with summary statistics
        """
        info = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        # Add sequence length stats if sequence column exists
        if "sequence" in df.columns:
            seq_lengths = df["sequence"].str.len()
            info["sequence_stats"] = {
                "min_length": int(seq_lengths.min()),
                "max_length": int(seq_lengths.max()),
                "mean_length": float(seq_lengths.mean()),
            }

        return info


def load_dataset(
    path: Union[str, Path],
    input_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience function to load a dataset.

    Args:
        path: Path to data file
        input_columns: Input column names
        target_column: Target column name
        nrows: Maximum rows to load

    Returns:
        DataFrame with loaded data
    """
    # Determine columns to load
    columns = None
    if input_columns or target_column:
        columns = []
        if input_columns:
            columns.extend(input_columns)
        if target_column:
            columns.append(target_column)
        # Also load common metadata columns if they exist
        # Don't specify these explicitly - let pandas load all columns
        # and we'll keep the metadata columns that are present
        columns.append("protein_id")

    # Load data without strict column requirements for metadata
    df = DataLoader.load(path, columns=None, nrows=nrows)

    # If specific columns were requested, validate they exist
    if columns:
        required = [col for col in columns if col in ["sequence", target_column] + (input_columns or [])]
        if required:
            DataLoader.validate_schema(df, required)

    # Validate required columns
    if input_columns:
        DataLoader.validate_schema(df, input_columns)
    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    return df
