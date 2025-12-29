#!/usr/bin/env python3
"""
Process AlphaFold PDB structures for secondary structure prediction dataset.

This script extracts protein sequences, pLDDT scores, and secondary structure
labels from AlphaFold PDB files.

NOTE: This version uses PLACEHOLDER secondary structure labels (all 'C' for coil).
For real secondary structure prediction, DSSP should be installed and integrated.
To install DSSP: conda install -c salilab dssp or apt-get install dssp

Input: ./mparsa/bench/swissprot_pdb/ (550,122 AlphaFold PDB files)
Output: ./mparsa/bench/procap-v2/data/structure/
  - secondary_structure_ss3.csv (10,000 samples)
  - secondary_structure_ss3_small.csv (1,000 samples)

Author: Generated for ProCap-v2 pipeline
Date: 2025-12-22
"""

import gzip
import os
import random
from pathlib import Path
import pandas as pd
from typing import Tuple, List, Optional
import sys

# Amino acid three-letter to one-letter code mapping
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Paths
INPUT_DIR = Path('./mparsa/bench/swissprot_pdb')
OUTPUT_DIR = Path('./mparsa/bench/procap-v2/data/structure')

# Processing parameters
FULL_DATASET_SIZE = 10000  # Number of structures to process for full dataset
SMALL_DATASET_SIZE = 1000  # Number of structures for small test dataset
MIN_SEQUENCE_LENGTH = 30
MIN_PLDDT = 50.0
PROGRESS_INTERVAL = 500


def extract_structure_info(pdb_path: Path) -> Optional[Tuple[str, List[float], List[Tuple[float, float, float]]]]:
    """
    Extract sequence, pLDDT scores, and CA coordinates from a PDB file.

    Args:
        pdb_path: Path to gzipped PDB file

    Returns:
        Tuple of (sequence, plddt_scores, ca_coordinates) or None if error
    """
    sequence = []
    plddt_scores = []
    coords = []  # CA coordinates for potential future SS calculation
    last_resnum = -1

    try:
        with gzip.open(pdb_path, 'rt') as f:
            for line in f:
                # Only process ATOM records for CA (C-alpha) atoms
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    resnum = int(line[22:26].strip())

                    # Skip if same residue (avoid duplicates)
                    if resnum == last_resnum:
                        continue

                    # Extract residue type, pLDDT (B-factor column), and coordinates
                    residue = line[17:20].strip()
                    plddt = float(line[60:66].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    # Map to one-letter code
                    aa = AA_MAP.get(residue, 'X')
                    sequence.append(aa)
                    plddt_scores.append(plddt)
                    coords.append((x, y, z))
                    last_resnum = resnum

        if not sequence:
            return None

        return ''.join(sequence), plddt_scores, coords

    except Exception as e:
        print(f"Error processing {pdb_path.name}: {e}", file=sys.stderr)
        return None


def assign_ss_placeholder(seq_len: int) -> str:
    """
    Assign placeholder secondary structure labels.

    NOTE: This is a PLACEHOLDER implementation. All positions are assigned 'C' (coil).
    For real secondary structure prediction, integrate DSSP:

    from Bio.PDB import PDBParser, DSSP
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    dssp = DSSP(structure[0], pdb_file)

    DSSP returns 8-class SS which should be mapped to 3-class:
    - H, G, I -> H (Helix: alpha-helix, 3-10 helix, pi-helix)
    - E, B -> E (Strand: beta-strand, beta-bridge)
    - T, S, - (or space) -> C (Coil: turn, bend, or random coil)

    Args:
        seq_len: Length of the sequence

    Returns:
        String of secondary structure labels (all 'C')
    """
    return 'C' * seq_len


def extract_uniprot_id(filename: str) -> str:
    """
    Extract UniProt ID from AlphaFold filename.

    Format: AF-{UniProt_ID}-F1-model_v6.pdb.gz
    Example: AF-A0A009IHW8-F1-model_v6.pdb.gz -> A0A009IHW8

    Args:
        filename: PDB filename

    Returns:
        UniProt ID
    """
    return filename.split('-')[1]


def process_pdb_files(pdb_files: List[Path], max_samples: int) -> pd.DataFrame:
    """
    Process PDB files and extract structure information.

    Args:
        pdb_files: List of PDB file paths
        max_samples: Maximum number of samples to collect

    Returns:
        DataFrame with protein_id, sequence, ss3_labels, plddt_mean
    """
    data = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Shuffle files for random sampling
    random.shuffle(pdb_files)

    print(f"Processing up to {max_samples} PDB files...")
    print(f"Total files available: {len(pdb_files)}")
    print(f"Filters: min_length={MIN_SEQUENCE_LENGTH}, min_plddt={MIN_PLDDT}")
    print()

    for i, pdb_path in enumerate(pdb_files):
        # Stop when we have enough samples
        if len(data) >= max_samples:
            break

        # Progress reporting
        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"Processed {i + 1} files | Collected {len(data)} samples | "
                  f"Skipped {skipped_count} | Errors {error_count}")

        # Extract structure information
        result = extract_structure_info(pdb_path)
        if result is None:
            error_count += 1
            continue

        sequence, plddt_scores, coords = result

        # Apply filters
        if len(sequence) < MIN_SEQUENCE_LENGTH:
            skipped_count += 1
            continue

        mean_plddt = sum(plddt_scores) / len(plddt_scores)
        if mean_plddt < MIN_PLDDT:
            skipped_count += 1
            continue

        # Extract UniProt ID
        protein_id = extract_uniprot_id(pdb_path.name)

        # Assign secondary structure (placeholder)
        ss3_labels = assign_ss_placeholder(len(sequence))

        # Add to dataset
        data.append({
            'protein_id': protein_id,
            'sequence': sequence,
            'ss3_labels': ss3_labels,
            'plddt_mean': round(mean_plddt, 2)
        })

        processed_count += 1

    print(f"\nProcessing complete!")
    print(f"Total files examined: {i + 1}")
    print(f"Samples collected: {len(data)}")
    print(f"Skipped (filters): {skipped_count}")
    print(f"Errors: {error_count}")

    return pd.DataFrame(data)


def print_dataset_statistics(df: pd.DataFrame, dataset_name: str):
    """Print statistics about the processed dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Number of proteins: {len(df)}")
    print(f"Sequence length - Min: {df['sequence'].str.len().min()}, "
          f"Max: {df['sequence'].str.len().max()}, "
          f"Mean: {df['sequence'].str.len().mean():.1f}")
    print(f"pLDDT - Min: {df['plddt_mean'].min():.2f}, "
          f"Max: {df['plddt_mean'].max():.2f}, "
          f"Mean: {df['plddt_mean'].mean():.2f}")

    # Check for unique sequences
    unique_seqs = df['sequence'].nunique()
    print(f"Unique sequences: {unique_seqs} ({unique_seqs/len(df)*100:.1f}%)")

    # Sample data
    print(f"\nSample entries:")
    print(df.head(3).to_string())


def main():
    """Main processing pipeline."""
    print("="*60)
    print("AlphaFold Secondary Structure Processing Pipeline")
    print("="*60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get all PDB files
    print("Scanning for PDB files...")
    pdb_files = list(INPUT_DIR.glob('AF-*.pdb.gz'))
    print(f"Found {len(pdb_files)} PDB files")

    if len(pdb_files) == 0:
        print("ERROR: No PDB files found!", file=sys.stderr)
        sys.exit(1)

    # Set random seed for reproducibility
    random.seed(42)

    # Process full dataset
    print(f"\n{'='*60}")
    print("Processing full dataset...")
    print(f"{'='*60}")
    df_full = process_pdb_files(pdb_files.copy(), FULL_DATASET_SIZE)

    if len(df_full) == 0:
        print("ERROR: No samples collected for full dataset!", file=sys.stderr)
        sys.exit(1)

    # Save full dataset
    full_output = OUTPUT_DIR / 'secondary_structure_ss3.csv'
    df_full.to_csv(full_output, index=False)
    print(f"\nSaved full dataset to: {full_output}")
    print_dataset_statistics(df_full, "Full Dataset")

    # Create small dataset (subsample from full)
    print(f"\n{'='*60}")
    print("Creating small dataset...")
    print(f"{'='*60}")
    df_small = df_full.sample(n=min(SMALL_DATASET_SIZE, len(df_full)), random_state=42)

    # Save small dataset
    small_output = OUTPUT_DIR / 'secondary_structure_ss3_small.csv'
    df_small.to_csv(small_output, index=False)
    print(f"\nSaved small dataset to: {small_output}")
    print_dataset_statistics(df_small, "Small Dataset")

    # Final summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Full dataset: {full_output}")
    print(f"  - {len(df_full)} proteins")
    print(f"Small dataset: {small_output}")
    print(f"  - {len(df_small)} proteins")
    print()
    print("NOTE: Secondary structure labels are PLACEHOLDERS (all 'C').")
    print("To generate real SS labels, install and integrate DSSP.")
    print("See script comments for implementation details.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
