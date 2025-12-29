#!/usr/bin/env python
"""
Process SCOP classification data for remote homology prediction.

This script parses SCOP classification data, extracts sequences from AlphaFold PDB files,
and creates datasets for superfamily classification tasks.
"""

import gzip
import os
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Amino acid mapping from 3-letter to 1-letter code
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def extract_sequence(pdb_path):
    """
    Extract protein sequence from AlphaFold PDB file.

    Args:
        pdb_path: Path to gzipped PDB file

    Returns:
        Protein sequence as a string
    """
    sequence = []
    last_resnum = -1

    try:
        with gzip.open(pdb_path, 'rt') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    resnum = int(line[22:26].strip())
                    if resnum != last_resnum:
                        residue = line[17:20].strip()
                        sequence.append(AA_MAP.get(residue, 'X'))
                        last_resnum = resnum
    except Exception as e:
        print(f"  Error reading {pdb_path}: {e}")
        return None

    return ''.join(sequence)

def parse_scopcla(scopcla):
    """
    Parse SCOPCLA string to extract class, fold, and superfamily values.

    Args:
        scopcla: String in format "TP=1,CL=1000003,CF=2000144,SF=3000034,FA=4000057"

    Returns:
        Dictionary with keys 'class', 'fold', 'superfamily'
    """
    result = {}
    for part in scopcla.split(','):
        key, value = part.split('=')
        if key == 'CL':
            result['class'] = value
        elif key == 'CF':
            result['fold'] = value
        elif key == 'SF':
            result['superfamily'] = value
    return result

def main():
    # Input/output paths
    scop_file = './mparsa/bench/scop-cla-latest.txt'
    pdb_dir = './mparsa/bench/swissprot_pdb'
    output_dir = './mparsa/bench/procap-v2/data/homology'

    print("Starting SCOP data processing...")
    print(f"Input SCOP file: {scop_file}")
    print(f"PDB directory: {pdb_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Storage for parsed data
    data_records = []
    superfamily_counts = defaultdict(int)

    # Statistics
    total_entries = 0
    processed_entries = 0
    missing_pdb = 0

    print("Parsing SCOP file and extracting sequences...")

    with open(scop_file, 'r') as f:
        for line in f:
            # Skip comment lines
            if line.startswith('#'):
                continue

            total_entries += 1

            # Progress reporting
            if total_entries % 1000 == 0:
                print(f"Processed {total_entries} entries, {processed_entries} sequences extracted, {missing_pdb} missing PDBs")

            # Parse line
            parts = line.strip().split()
            if len(parts) < 11:
                continue

            # Extract FA-UNIID (UniProt ID) and SCOPCLA
            fa_uniid = parts[3]
            scopcla = parts[10]

            # Parse SCOPCLA to get class, fold, superfamily
            scop_data = parse_scopcla(scopcla)

            # Check if all required fields are present
            if not all(k in scop_data for k in ['class', 'fold', 'superfamily']):
                continue

            # Construct PDB file path
            pdb_path = os.path.join(pdb_dir, f'AF-{fa_uniid}-F1-model_v6.pdb.gz')

            # Check if PDB file exists
            if not os.path.exists(pdb_path):
                missing_pdb += 1
                continue

            # Extract sequence from PDB
            sequence = extract_sequence(pdb_path)
            if sequence is None or len(sequence) == 0:
                continue

            # Store record
            record = {
                'protein_id': fa_uniid,
                'sequence': sequence,
                'superfamily': scop_data['superfamily'],
                'fold': scop_data['fold'],
                'class': scop_data['class']
            }
            data_records.append(record)
            superfamily_counts[scop_data['superfamily']] += 1
            processed_entries += 1

    print()
    print(f"Finished parsing. Total entries: {total_entries}")
    print(f"Sequences extracted: {processed_entries}")
    print(f"Missing PDB files: {missing_pdb}")
    print()

    # Create DataFrame
    df = pd.DataFrame(data_records)
    print(f"Created DataFrame with {len(df)} entries")
    print()

    # Filter to superfamilies with >= 5 members
    print("Filtering to superfamilies with >= 5 members...")
    superfamilies_to_keep = {sf for sf, count in superfamily_counts.items() if count >= 5}
    print(f"Superfamilies before filtering: {len(superfamily_counts)}")
    print(f"Superfamilies after filtering (>= 5 members): {len(superfamilies_to_keep)}")

    df_filtered = df[df['superfamily'].isin(superfamilies_to_keep)]
    print(f"Entries after filtering: {len(df_filtered)}")
    print()

    # Print some statistics
    print("Dataset statistics:")
    print(f"  Total proteins: {len(df_filtered)}")
    print(f"  Unique superfamilies: {df_filtered['superfamily'].nunique()}")
    print(f"  Unique folds: {df_filtered['fold'].nunique()}")
    print(f"  Unique classes: {df_filtered['class'].nunique()}")
    print(f"  Average sequence length: {df_filtered['sequence'].str.len().mean():.1f}")
    print(f"  Min sequence length: {df_filtered['sequence'].str.len().min()}")
    print(f"  Max sequence length: {df_filtered['sequence'].str.len().max()}")
    print()

    # Save full dataset
    output_file = os.path.join(output_dir, 'remote_homology.csv')
    df_filtered.to_csv(output_file, index=False)
    print(f"Saved full dataset to: {output_file}")

    # Create and save small dataset (1000 samples)
    if len(df_filtered) >= 1000:
        df_small = df_filtered.sample(n=1000, random_state=42)
        output_file_small = os.path.join(output_dir, 'remote_homology_small.csv')
        df_small.to_csv(output_file_small, index=False)
        print(f"Saved small dataset (1000 samples) to: {output_file_small}")
    else:
        print(f"Dataset has only {len(df_filtered)} entries, not creating small dataset")

    print()
    print("Processing complete!")

if __name__ == '__main__':
    main()
