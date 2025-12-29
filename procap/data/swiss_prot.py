"""
Swiss-Prot specific data loading utilities.

Provides efficient loading and processing of Swiss-Prot protein data.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd


class SwissProtLoader:
    """
    Efficient loader for Swiss-Prot parquet files.

    Handles:
    - Multiple parquet file shards
    - Column selection for memory efficiency
    - GO term parsing
    - Filtering by species/family

    Usage:
        loader = SwissProtLoader(Path("data/"))
        df = loader.load_all(max_rows=1000)
        stats = loader.get_statistics()
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        pattern: str = "swiss_above_20_swissprot_*.parquet",
    ):
        """
        Initialize Swiss-Prot loader.

        Args:
            data_dir: Directory containing parquet files
            pattern: Glob pattern for parquet files
        """
        self.data_dir = Path(data_dir)
        self.pattern = pattern
        self._parquet_files: Optional[List[Path]] = None

    @property
    def parquet_files(self) -> List[Path]:
        """Get list of parquet files."""
        if self._parquet_files is None:
            self._parquet_files = sorted(self.data_dir.glob(self.pattern))
            if not self._parquet_files:
                raise FileNotFoundError(
                    f"No parquet files matching {self.pattern} in {self.data_dir}"
                )
        return self._parquet_files

    def load_all(
        self,
        columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load all parquet files into a single DataFrame.

        Args:
            columns: Columns to load (None for all)
            max_rows: Maximum number of rows to load

        Returns:
            DataFrame with all data
        """
        dfs = []
        total_rows = 0

        for pq_file in self.parquet_files:
            df = pd.read_parquet(pq_file, columns=columns)

            if max_rows and total_rows + len(df) > max_rows:
                df = df.iloc[: max_rows - total_rows]
                dfs.append(df)
                break

            dfs.append(df)
            total_rows += len(df)

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def iter_batches(
        self,
        batch_size: int = 1000,
        columns: Optional[List[str]] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over data in batches for memory efficiency.

        Args:
            batch_size: Number of rows per batch
            columns: Columns to load

        Yields:
            DataFrame batches
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            # Fallback to pandas if pyarrow not available
            for pq_file in self.parquet_files:
                df = pd.read_parquet(pq_file, columns=columns)
                for start in range(0, len(df), batch_size):
                    yield df.iloc[start : start + batch_size]
            return

        for pq_file in self.parquet_files:
            table = pq.read_table(pq_file, columns=columns)
            for batch in table.to_batches(max_chunksize=batch_size):
                yield batch.to_pandas()

    @staticmethod
    def parse_go_terms(go_string: str) -> List[str]:
        """
        Parse semicolon-separated GO terms.

        Args:
            go_string: GO terms as "GO:0001;GO:0002;..."

        Returns:
            List of GO term IDs
        """
        if pd.isna(go_string) or not go_string:
            return []
        return [t.strip() for t in go_string.split(";") if t.strip()]

    def filter_by_species(
        self,
        species: str,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Filter data by species.

        Args:
            species: Species name to filter
            df: DataFrame to filter (loads all if None)

        Returns:
            Filtered DataFrame
        """
        if df is None:
            df = self.load_all()
        return df[df["species"] == species].copy()

    def filter_by_family(
        self,
        family: str,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Filter data by protein family.

        Args:
            family: Family ID to filter
            df: DataFrame to filter (loads all if None)

        Returns:
            Filtered DataFrame
        """
        if df is None:
            df = self.load_all()
        return df[df["family"] == family].copy()

    def filter_by_sequence_length(
        self,
        min_length: int = 0,
        max_length: int = 10000,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Filter data by sequence length.

        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            df: DataFrame to filter (loads all if None)

        Returns:
            Filtered DataFrame
        """
        if df is None:
            df = self.load_all(columns=["sequence"])

        seq_lengths = df["sequence"].str.len()
        mask = (seq_lengths >= min_length) & (seq_lengths <= max_length)
        return df[mask].copy()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with summary statistics
        """
        # Load minimal columns for stats
        df = self.load_all(columns=["protein_id", "sequence", "go_terms", "family", "species"])

        stats = {
            "total_proteins": len(df),
            "num_files": len(self.parquet_files),
        }

        if "species" in df.columns:
            stats["unique_species"] = df["species"].nunique()

        if "family" in df.columns:
            stats["unique_families"] = df["family"].nunique()

        if "sequence" in df.columns:
            seq_lengths = df["sequence"].str.len()
            stats["sequence_length"] = {
                "min": int(seq_lengths.min()),
                "max": int(seq_lengths.max()),
                "mean": float(seq_lengths.mean()),
                "median": float(seq_lengths.median()),
            }

        if "go_terms" in df.columns:
            go_counts = df["go_terms"].apply(
                lambda x: len(self.parse_go_terms(x)) if pd.notna(x) else 0
            )
            stats["go_terms_per_protein"] = {
                "min": int(go_counts.min()),
                "max": int(go_counts.max()),
                "mean": float(go_counts.mean()),
            }

        return stats

    def create_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_by: Optional[str] = None,
    ) -> tuple:
        """
        Create train/test split of the data.

        Args:
            test_size: Fraction of data for test set
            random_state: Random seed
            stratify_by: Column to stratify by

        Returns:
            (train_df, test_df) tuple
        """
        from sklearn.model_selection import train_test_split

        df = self.load_all()

        stratify = df[stratify_by] if stratify_by and stratify_by in df.columns else None

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        return train_df, test_df
