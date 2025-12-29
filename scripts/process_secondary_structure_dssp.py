#!/usr/bin/env python3
"""
Process AlphaFold structures to extract secondary structure labels using pydssp.

This script:
1. Reads gzipped PDB files from AlphaFold
2. Uses pydssp (Python DSSP implementation) to assign secondary structure
3. Outputs CSV with protein_id, sequence, ss3_labels, ss8_labels, plddt_mean
"""

import gzip
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from Bio.PDB import PDBParser
import pydssp

# Configuration
PDB_DIR = Path("./mparsa/bench/swissprot_pdb")
OUTPUT_DIR = Path("./mparsa/bench/procap-v2/data/structure")

# Sample sizes
FULL_SAMPLE_SIZE = 10000
SMALL_SAMPLE_SIZE = 1000

# Quality filters
MIN_SEQ_LENGTH = 30
MIN_PLDDT = 50.0

# Amino acid mapping (3-letter to 1-letter)
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# DSSP SS8 to SS3 mapping
SS8_TO_SS3 = {
    'H': 'H',  # Alpha helix
    'G': 'H',  # 3-10 helix
    'I': 'H',  # Pi helix
    'E': 'E',  # Extended strand
    'B': 'E',  # Beta bridge
    'T': 'C',  # Turn
    'S': 'C',  # Bend
    ' ': 'C',  # Coil/Loop
    '-': 'C',  # Unknown/Coil
    'C': 'C',  # Coil
}

# Initialize PDB parser
pdb_parser = PDBParser(QUIET=True)


def extract_backbone_coords(pdb_path: Path) -> Tuple[Optional[np.ndarray], str, List[float]]:
    """
    Extract backbone coordinates (N, CA, C, O), sequence, and pLDDT scores from PDB.

    Returns:
        (coords array [N, 4, 3], sequence string, plddt_scores list)
        Returns (None, "", []) if extraction fails
    """
    try:
        # Uncompress if needed
        opener = gzip.open if str(pdb_path).endswith('.gz') else open

        with opener(pdb_path, 'rt') as f:
            pdb_content = f.read()

        # Write to temp file for BioPython
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp.write(pdb_content)
            tmp_path = tmp.name

        # Parse structure
        structure = pdb_parser.get_structure('protein', tmp_path)
        os.unlink(tmp_path)

        model = structure[0]
        chain = list(model.get_chains())[0]

        # Extract backbone coordinates and sequence
        coords = []
        sequence = []
        plddt_scores = []

        for residue in chain.get_residues():
            # Skip heteroatoms
            if residue.id[0] != ' ':
                continue

            resname = residue.get_resname()
            aa = AA_MAP.get(resname, 'X')

            # Skip unknown amino acids
            if aa == 'X':
                continue

            try:
                n = residue['N'].get_coord()
                ca = residue['CA'].get_coord()
                c = residue['C'].get_coord()
                o = residue['O'].get_coord()

                # Get pLDDT from B-factor of CA
                plddt = residue['CA'].get_bfactor()

                coords.append([n, ca, c, o])
                sequence.append(aa)
                plddt_scores.append(plddt)

            except KeyError:
                # Missing backbone atoms, skip this residue
                continue

        if len(coords) == 0:
            return None, "", []

        return np.array(coords, dtype=np.float32), ''.join(sequence), plddt_scores

    except Exception as e:
        return None, "", []


def assign_secondary_structure(coords: np.ndarray) -> Tuple[str, str]:
    """
    Assign secondary structure using pydssp.

    Args:
        coords: Backbone coordinates [N, 4, 3] for N, CA, C, O atoms

    Returns:
        (ss8_string, ss3_string)
    """
    # Convert to torch tensor and add batch dimension
    coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

    # Run pydssp
    ss8_array = pydssp.assign(coords_tensor)

    # Convert to strings
    ss8_str = ''.join(ss8_array[0])
    ss3_str = ''.join([SS8_TO_SS3.get(s, 'C') for s in ss8_str])

    return ss8_str, ss3_str


def process_pdb_file(pdb_path: Path) -> Optional[Dict]:
    """
    Process a single PDB file to extract structure info and SS labels.

    Returns:
        Dictionary with protein_id, sequence, ss3_labels, ss8_labels, plddt_mean
        or None if processing failed
    """
    try:
        # Extract UniProt ID from filename: AF-{UNIPROT}-F1-model_v6.pdb.gz
        filename = pdb_path.name
        if filename.startswith('AF-') and '-F1-model' in filename:
            protein_id = filename.split('-')[1]
        else:
            protein_id = filename.replace('.pdb.gz', '').replace('.pdb', '')

        # Extract backbone coordinates, sequence, and pLDDT
        coords, sequence, plddt_scores = extract_backbone_coords(pdb_path)

        if coords is None or len(sequence) == 0:
            return None

        # Quality filters
        if len(sequence) < MIN_SEQ_LENGTH:
            return None

        mean_plddt = sum(plddt_scores) / len(plddt_scores)
        if mean_plddt < MIN_PLDDT:
            return None

        # Assign secondary structure using pydssp
        ss8_str, ss3_str = assign_secondary_structure(coords)

        # Verify lengths match
        if len(ss3_str) != len(sequence):
            return None

        return {
            'protein_id': protein_id,
            'sequence': sequence,
            'ss3_labels': ss3_str,
            'ss8_labels': ss8_str,
            'plddt_mean': round(mean_plddt, 2)
        }

    except Exception as e:
        return None


def main():
    print("=" * 60)
    print("Secondary Structure Processing with pydssp")
    print("=" * 60)

    print(f"Input directory: {PDB_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using pydssp (Python DSSP implementation)")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # List all PDB files
    print("Scanning PDB files...")
    pdb_files = list(PDB_DIR.glob("AF-*-F1-model_v*.pdb.gz"))
    print(f"Found {len(pdb_files)} PDB files")

    # Random sample for processing
    random.seed(42)
    sample_files = random.sample(pdb_files, min(len(pdb_files), FULL_SAMPLE_SIZE * 2))
    print(f"Sampling {len(sample_files)} files for processing")
    print()

    # Process files
    records = []
    failed = 0

    print("Processing PDB files with pydssp...")
    for pdb_path in tqdm(sample_files, desc="Processing"):
        result = process_pdb_file(pdb_path)
        if result:
            records.append(result)
            if len(records) >= FULL_SAMPLE_SIZE:
                break
        else:
            failed += 1

    print()
    print(f"Successfully processed: {len(records)}")
    print(f"Failed: {failed}")

    if not records:
        print("ERROR: No records processed successfully")
        return

    # Create DataFrame
    df = pd.DataFrame(records)

    # Statistics
    print()
    print("Dataset Statistics:")
    print(f"  Total proteins: {len(df)}")
    print(f"  Sequence length: min={df['sequence'].str.len().min()}, "
          f"max={df['sequence'].str.len().max()}, "
          f"mean={df['sequence'].str.len().mean():.1f}")
    print(f"  pLDDT: min={df['plddt_mean'].min():.1f}, "
          f"max={df['plddt_mean'].max():.1f}, "
          f"mean={df['plddt_mean'].mean():.1f}")

    # SS3 class distribution
    all_ss3 = ''.join(df['ss3_labels'].tolist())
    h_count = all_ss3.count('H')
    e_count = all_ss3.count('E')
    c_count = all_ss3.count('C')
    total = len(all_ss3)

    print()
    print("SS3 Class Distribution:")
    print(f"  H (Helix):  {h_count:,} ({100*h_count/total:.1f}%)")
    print(f"  E (Strand): {e_count:,} ({100*e_count/total:.1f}%)")
    print(f"  C (Coil):   {c_count:,} ({100*c_count/total:.1f}%)")

    # Save full dataset
    full_path = OUTPUT_DIR / "secondary_structure_ss3.csv"
    df.to_csv(full_path, index=False)
    print()
    print(f"Saved full dataset: {full_path}")
    print(f"  Size: {full_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save small dataset
    small_df = df.sample(n=min(SMALL_SAMPLE_SIZE, len(df)), random_state=42)
    small_path = OUTPUT_DIR / "secondary_structure_ss3_small.csv"
    small_df.to_csv(small_path, index=False)
    print(f"Saved small dataset: {small_path}")
    print(f"  Size: {small_path.stat().st_size / 1024:.1f} KB")

    # Verify a sample
    print()
    print("Sample record:")
    sample = df.iloc[0]
    print(f"  protein_id: {sample['protein_id']}")
    print(f"  sequence[:50]: {sample['sequence'][:50]}...")
    print(f"  ss3_labels[:50]: {sample['ss3_labels'][:50]}...")
    print(f"  ss8_labels[:50]: {sample['ss8_labels'][:50]}...")
    print(f"  plddt_mean: {sample['plddt_mean']}")

    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
