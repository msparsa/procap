#!/usr/bin/env python3
"""
Process ProteinGym variant effect data and save to procap-v2 benchmark.

This script processes all CSV files from ProteinGym DMS substitutions data,
extracting wild-type sequences and variant information to create a standardized
variant effect prediction dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re


def parse_variant(variant_str):
    """
    Parse variant notation (e.g., 'G126S' or 'G126S:N139D').
    Returns list of tuples: [(wt_aa, pos, mut_aa), ...]
    """
    variants = []
    for var in variant_str.split(':'):
        var = var.strip()
        # Match pattern: WT_AA + Position + MUT_AA
        match = re.match(r'([A-Z])(\d+)([A-Z])', var)
        if match:
            wt_aa, pos, mut_aa = match.groups()
            variants.append((wt_aa, int(pos) - 1, mut_aa))  # Convert to 0-indexed
        else:
            raise ValueError(f"Invalid variant notation: {var}")
    return variants


def reconstruct_wildtype_sequence(mutated_sequence, variant_str):
    """
    Reconstruct wild-type sequence from mutated sequence and variant notation.
    """
    # Parse the variant(s)
    variants = parse_variant(variant_str)

    # Start with mutated sequence
    wt_seq = list(mutated_sequence)

    # Reverse the mutations
    for wt_aa, pos, mut_aa in variants:
        # Verify that the mutated sequence has the expected mutant amino acid
        if pos < len(wt_seq):
            if wt_seq[pos] != mut_aa:
                # This might happen with multi-mutants or data inconsistencies
                # We'll still proceed but use the variant notation as ground truth
                pass
            wt_seq[pos] = wt_aa

    return ''.join(wt_seq)


def validate_amino_acids(sequence):
    """
    Check if sequence contains only valid amino acids.
    """
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa in valid_aas for aa in sequence)


def extract_protein_id(filename):
    """
    Extract protein ID from filename.
    E.g., 'PABP_YEAST_Melamed_2013.csv' -> 'PABP_YEAST'
    """
    # Remove .csv extension
    name = filename.replace('.csv', '')
    # Split by underscore and take first two parts (typically PROTEIN_ORGANISM)
    parts = name.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def process_file(file_path):
    """
    Process a single ProteinGym CSV file.
    Returns DataFrame with processed data.
    """
    filename = os.path.basename(file_path)
    protein_id = extract_protein_id(filename)

    # Read CSV
    df = pd.read_csv(file_path)

    # Check required columns
    required_cols = ['mutant', 'mutated_sequence', 'DMS_score', 'DMS_score_bin']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: {filename} missing required columns, skipping")
        return None

    # Get wild-type sequence from first variant
    # We'll use the first row to reconstruct the wild-type
    first_variant = df.iloc[0]
    try:
        wt_sequence = reconstruct_wildtype_sequence(
            first_variant['mutated_sequence'],
            first_variant['mutant']
        )
    except Exception as e:
        print(f"Warning: Could not reconstruct wild-type for {filename}: {e}")
        return None

    # Validate wild-type sequence
    if not validate_amino_acids(wt_sequence):
        print(f"Warning: {filename} contains invalid amino acids, skipping")
        return None

    # Process each row
    processed_rows = []
    for idx, row in df.iterrows():
        try:
            processed_row = {
                'protein_id': protein_id,
                'sequence': wt_sequence,
                'variant': row['mutant'],
                'mutated_sequence': row['mutated_sequence'],
                'effect_score': row['DMS_score'],
                'effect_bin': int(row['DMS_score_bin']),
                'source_file': filename
            }
            processed_rows.append(processed_row)
        except Exception as e:
            print(f"Warning: Error processing row {idx} in {filename}: {e}")
            continue

    if not processed_rows:
        return None

    return pd.DataFrame(processed_rows)


def main():
    """
    Main processing function.
    """
    # Define paths
    input_dir = Path('./mparsa/bench/ProteinGYM/DMS_ProteinGym_substitutions')
    output_dir = Path('./mparsa/bench/procap-v2/data/variant_effect')

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all CSV files
    csv_files = sorted(list(input_dir.glob('*.csv')))
    print(f"Found {len(csv_files)} CSV files to process")

    # Process all files
    all_data = []
    successful = 0
    failed = 0

    for i, file_path in enumerate(csv_files, 1):
        try:
            df = process_file(file_path)
            if df is not None:
                all_data.append(df)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            failed += 1

        # Print progress every 50 files
        if i % 50 == 0:
            print(f"Processed {i}/{len(csv_files)} files ({successful} successful, {failed} failed)")

    print(f"\nProcessing complete: {successful} successful, {failed} failed")

    if not all_data:
        print("No data processed successfully. Exiting.")
        return

    # Concatenate all DataFrames
    print("\nConcatenating all data...")
    full_df = pd.concat(all_data, ignore_index=True)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total records: {len(full_df):,}")
    print(f"Unique proteins: {full_df['protein_id'].nunique()}")
    print(f"Unique source files: {full_df['source_file'].nunique()}")
    print(f"\nEffect score statistics:")
    print(full_df['effect_score'].describe())
    print(f"\nEffect bin distribution:")
    print(full_df['effect_bin'].value_counts().sort_index())
    print(f"\nRecords per protein (top 10):")
    print(full_df['protein_id'].value_counts().head(10))

    # Save full dataset
    full_output = output_dir / 'proteingym_substitutions.csv'
    print(f"\nSaving full dataset to: {full_output}")
    full_df.to_csv(full_output, index=False)
    print(f"Saved {len(full_df):,} records")

    # Create small sample (1000 random samples)
    print("\nCreating small sample dataset...")
    if len(full_df) > 1000:
        small_df = full_df.sample(n=1000, random_state=42)
    else:
        small_df = full_df.copy()

    small_output = output_dir / 'proteingym_substitutions_small.csv'
    print(f"Saving small dataset to: {small_output}")
    small_df.to_csv(small_output, index=False)
    print(f"Saved {len(small_df):,} records")

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
