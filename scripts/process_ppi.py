#!/usr/bin/env python3
"""
Process STRING protein-protein interaction data.

This script:
1. Extracts human-human interactions from STRING file
2. Filters high-confidence interactions (score >= 700)
3. Maps Ensembl IDs to UniProt IDs using mygene
4. Extracts sequences from AlphaFold PDB files
5. Creates balanced positive/negative datasets
6. Outputs CSV files for training
"""

import gzip
import csv
import os
import random
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List
import sys

# Configuration
STRING_FILE = "./mparsa/bench/STRING.protein.physical.links.v12.0.txt.gz"
PDB_DIR = "./mparsa/bench/swissprot_pdb/"
OUTPUT_DIR = "./mparsa/bench/procap-v2/data/interaction/"
TEMP_HUMAN_FILE = os.path.join(OUTPUT_DIR, "temp_human_interactions.txt.gz")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ppi_prediction.csv")
OUTPUT_FILE_SMALL = os.path.join(OUTPUT_DIR, "ppi_prediction_small.csv")

# Parameters
HUMAN_TAX_ID = "9606"
MIN_SCORE = 700
MAX_POSITIVE_PAIRS = 25000  # Limit positive pairs
TARGET_PAIRS = 50000  # Total pairs (25K positive + 25K negative)
SMALL_SAMPLE_SIZE = 1000

random.seed(42)


def extract_human_interactions():
    """Extract human interactions from STRING file using grep (much faster)."""
    print("Extracting human-human interactions from STRING file...")
    print("This may take several minutes...")

    # Use grep to extract human interactions - much faster than Python
    cmd = f'gunzip -c {STRING_FILE} | grep "^9606\\." | grep " 9606\\." | gzip > {TEMP_HUMAN_FILE}'

    subprocess.run(cmd, shell=True, check=True)

    # Count lines
    count_cmd = f'gunzip -c {TEMP_HUMAN_FILE} | wc -l'
    result = subprocess.run(count_cmd, shell=True, capture_output=True, text=True)
    num_interactions = int(result.stdout.strip())

    print(f"Extracted {num_interactions:,} human-human interactions")
    return num_interactions


def get_available_uniprot_ids(pdb_dir: str) -> Set[str]:
    """Extract UniProt IDs from AlphaFold PDB filenames."""
    print("Building set of available UniProt IDs from AlphaFold PDB files...")
    uniprot_ids = set()

    pattern = re.compile(r'AF-([A-Z0-9]+)-F1-model_v6\.pdb\.gz')

    for filename in os.listdir(pdb_dir):
        match = pattern.match(filename)
        if match:
            uniprot_ids.add(match.group(1))

    print(f"Found {len(uniprot_ids)} UniProt IDs with AlphaFold structures")
    return uniprot_ids


def extract_sequence_from_pdb(pdb_file: str) -> str:
    """Extract amino acid sequence from a gzipped PDB file."""
    sequence = []

    with gzip.open(pdb_file, 'rt') as f:
        for line in f:
            if line.startswith('ATOM'):
                # PDB ATOM records contain residue information
                # Columns 18-20 contain the residue name
                # Column 23-26 contain the residue sequence number
                residue_name = line[17:20].strip()
                residue_num = int(line[22:26].strip())

                # Map 3-letter codes to 1-letter codes
                aa_map = {
                    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                }

                if residue_name in aa_map:
                    # Store with residue number to handle insertions
                    if not sequence or sequence[-1][0] != residue_num:
                        sequence.append((residue_num, aa_map[residue_name]))

    # Extract just the amino acids in order
    return ''.join([aa for _, aa in sequence])


def map_ensembl_to_uniprot_batch(ensembl_ids: Set[str]) -> Dict[str, str]:
    """Map Ensembl protein IDs to UniProt IDs using mygene."""
    print(f"Mapping {len(ensembl_ids)} Ensembl IDs to UniProt IDs...")

    try:
        import mygene
        mg = mygene.MyGeneInfo()

        # Query in batches
        ensembl_list = list(ensembl_ids)
        batch_size = 1000
        mapping = {}

        for i in range(0, len(ensembl_list), batch_size):
            batch = ensembl_list[i:i+batch_size]
            if i % 10000 == 0:
                print(f"  Processing batch {i//batch_size + 1}/{(len(ensembl_list) + batch_size - 1)//batch_size}")

            results = mg.querymany(
                batch,
                scopes='ensembl.protein',
                fields='uniprot.Swiss-Prot',
                species='human',
                returnall=True
            )

            for result in results['out']:
                if 'uniprot' in result and 'Swiss-Prot' in result['uniprot']:
                    ensembl_id = result['query']
                    uniprot_data = result['uniprot']['Swiss-Prot']

                    # Handle both single string and list of strings
                    if isinstance(uniprot_data, str):
                        mapping[ensembl_id] = uniprot_data
                    elif isinstance(uniprot_data, list) and len(uniprot_data) > 0:
                        mapping[ensembl_id] = uniprot_data[0]  # Take first match

        print(f"Successfully mapped {len(mapping)} Ensembl IDs to UniProt")
        return mapping

    except ImportError:
        print("ERROR: mygene package not installed. Installing...")
        os.system(f"{sys.executable} -m pip install mygene")
        # Retry after installation
        return map_ensembl_to_uniprot_batch(ensembl_ids)


def process_human_interactions(human_file: str) -> Tuple[List[Tuple], Set[str]]:
    """
    Process human interactions file to extract high-confidence PPIs.
    Returns: (positive_pairs, all_ensembl_proteins)
    """
    print(f"Processing human interactions file")
    print(f"Filtering for score >= {MIN_SCORE}")

    positive_pairs = []
    all_ensembl_proteins = set()
    lines_processed = 0
    high_conf_pairs = 0

    with gzip.open(human_file, 'rt') as f:
        for line in f:
            lines_processed += 1

            if lines_processed % 100000 == 0:
                print(f"  Processed {lines_processed:,} lines, {high_conf_pairs:,} high-confidence")

            parts = line.strip().split()
            if len(parts) != 3:
                continue

            protein1, protein2, score_str = parts
            score = int(score_str)

            # Filter for high confidence
            if score < MIN_SCORE:
                continue

            high_conf_pairs += 1

            # Extract Ensembl IDs (remove taxonomy prefix)
            ensembl1 = protein1.split('.', 1)[1]
            ensembl2 = protein2.split('.', 1)[1]

            all_ensembl_proteins.add(ensembl1)
            all_ensembl_proteins.add(ensembl2)

            positive_pairs.append((ensembl1, ensembl2, score))

            # Limit the number of positive pairs
            if len(positive_pairs) >= MAX_POSITIVE_PAIRS * 3:  # Collect more than needed
                break

    print(f"Completed processing {lines_processed:,} lines")
    print(f"Found {high_conf_pairs:,} high-confidence pairs")
    print(f"Collected {len(positive_pairs):,} positive pairs for processing")

    return positive_pairs, all_ensembl_proteins


def create_negative_pairs(
    all_proteins: List[str],
    positive_pairs_set: Set[Tuple[str, str]],
    num_negative: int
) -> List[Tuple[str, str, int]]:
    """Create negative pairs that don't exist in STRING."""
    print(f"Creating {num_negative:,} negative pairs...")

    negative_pairs = []
    attempts = 0
    max_attempts = num_negative * 10

    while len(negative_pairs) < num_negative and attempts < max_attempts:
        attempts += 1

        # Random pair
        p1, p2 = random.sample(all_proteins, 2)

        # Ensure canonical ordering
        if p1 > p2:
            p1, p2 = p2, p1

        # Check if it's not a positive pair
        if (p1, p2) not in positive_pairs_set and (p2, p1) not in positive_pairs_set:
            negative_pairs.append((p1, p2, 0))  # Score 0 for negative pairs

        if len(negative_pairs) % 5000 == 0 and len(negative_pairs) > 0:
            print(f"  Created {len(negative_pairs):,} negative pairs")

    print(f"Created {len(negative_pairs):,} negative pairs in {attempts:,} attempts")
    return negative_pairs


def build_dataset(
    positive_pairs: List[Tuple],
    negative_pairs: List[Tuple],
    ensembl_to_uniprot: Dict[str, str],
    available_uniprot_ids: Set[str],
    pdb_dir: str
) -> List[Dict]:
    """Build the final dataset with sequences."""
    print("Building final dataset with sequences...")

    dataset = []
    sequence_cache = {}

    all_pairs = [(p1, p2, score, 1) for p1, p2, score in positive_pairs] + \
                [(p1, p2, score, 0) for p1, p2, score in negative_pairs]

    for idx, (ensembl1, ensembl2, score, interaction) in enumerate(all_pairs):
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,}/{len(all_pairs):,} pairs, {len(dataset):,} valid")

        # Map to UniProt
        uniprot1 = ensembl_to_uniprot.get(ensembl1)
        uniprot2 = ensembl_to_uniprot.get(ensembl2)

        # Check if both have UniProt IDs and AlphaFold structures
        if not uniprot1 or not uniprot2:
            continue

        if uniprot1 not in available_uniprot_ids or uniprot2 not in available_uniprot_ids:
            continue

        # Get sequences
        if uniprot1 not in sequence_cache:
            pdb_file = os.path.join(pdb_dir, f"AF-{uniprot1}-F1-model_v6.pdb.gz")
            try:
                sequence_cache[uniprot1] = extract_sequence_from_pdb(pdb_file)
            except Exception as e:
                print(f"  Warning: Failed to extract sequence for {uniprot1}: {e}")
                continue

        if uniprot2 not in sequence_cache:
            pdb_file = os.path.join(pdb_dir, f"AF-{uniprot2}-F1-model_v6.pdb.gz")
            try:
                sequence_cache[uniprot2] = extract_sequence_from_pdb(pdb_file)
            except Exception as e:
                print(f"  Warning: Failed to extract sequence for {uniprot2}: {e}")
                continue

        seq1 = sequence_cache[uniprot1]
        seq2 = sequence_cache[uniprot2]

        if not seq1 or not seq2:
            continue

        dataset.append({
            'protein_id_a': uniprot1,
            'sequence_a': seq1,
            'protein_id_b': uniprot2,
            'sequence_b': seq2,
            'interaction': interaction,
            'combined_score': score
        })

    print(f"Final dataset contains {len(dataset):,} pairs")
    return dataset


def save_dataset(dataset: List[Dict], output_file: str, sample_size: int = None):
    """Save dataset to CSV file."""
    if sample_size:
        dataset = random.sample(dataset, min(sample_size, len(dataset)))

    print(f"Saving {len(dataset):,} pairs to {output_file}")

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['protein_id_a', 'sequence_a', 'protein_id_b', 'sequence_b', 'interaction', 'combined_score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"Saved to {output_file}")


def main():
    """Main processing pipeline."""
    print("=" * 80)
    print("STRING Protein-Protein Interaction Dataset Builder")
    print("=" * 80)

    # Step 1: Extract human interactions (if not already done)
    if not os.path.exists(TEMP_HUMAN_FILE):
        extract_human_interactions()
    else:
        print(f"Using existing human interactions file: {TEMP_HUMAN_FILE}")

    # Step 2: Get available UniProt IDs
    available_uniprot_ids = get_available_uniprot_ids(PDB_DIR)

    # Step 3: Process human interactions
    positive_pairs, all_ensembl_proteins = process_human_interactions(TEMP_HUMAN_FILE)

    # Step 4: Map Ensembl to UniProt
    ensembl_to_uniprot = map_ensembl_to_uniprot_batch(all_ensembl_proteins)

    # Step 5: Filter positive pairs to those with both proteins available
    print("Filtering positive pairs to those with available structures...")
    filtered_positive = []
    positive_pairs_set = set()

    for ensembl1, ensembl2, score in positive_pairs:
        uniprot1 = ensembl_to_uniprot.get(ensembl1)
        uniprot2 = ensembl_to_uniprot.get(ensembl2)

        if uniprot1 and uniprot2 and uniprot1 in available_uniprot_ids and uniprot2 in available_uniprot_ids:
            # Canonical ordering
            if uniprot1 > uniprot2:
                uniprot1, uniprot2 = uniprot2, uniprot1
                ensembl1, ensembl2 = ensembl2, ensembl1

            filtered_positive.append((ensembl1, ensembl2, score))
            positive_pairs_set.add((uniprot1, uniprot2))

    print(f"Filtered to {len(filtered_positive):,} positive pairs with available structures")

    # Limit to target
    if len(filtered_positive) > MAX_POSITIVE_PAIRS:
        print(f"Sampling {MAX_POSITIVE_PAIRS:,} from {len(filtered_positive):,} positive pairs")
        filtered_positive = random.sample(filtered_positive, MAX_POSITIVE_PAIRS)
        # Rebuild positive_pairs_set
        positive_pairs_set = set()
        for ensembl1, ensembl2, score in filtered_positive:
            uniprot1 = ensembl_to_uniprot.get(ensembl1)
            uniprot2 = ensembl_to_uniprot.get(ensembl2)
            if uniprot1 > uniprot2:
                uniprot1, uniprot2 = uniprot2, uniprot1
            positive_pairs_set.add((uniprot1, uniprot2))

    # Step 6: Get proteins that have structures
    proteins_with_structures = []
    for ensembl_id in all_ensembl_proteins:
        uniprot_id = ensembl_to_uniprot.get(ensembl_id)
        if uniprot_id and uniprot_id in available_uniprot_ids:
            proteins_with_structures.append(ensembl_id)

    print(f"Found {len(proteins_with_structures):,} proteins with available structures")

    # Step 7: Create negative pairs
    negative_pairs = create_negative_pairs(
        proteins_with_structures,
        positive_pairs_set,
        len(filtered_positive)  # Same number as positive
    )

    # Step 8: Build final dataset with sequences
    dataset = build_dataset(
        filtered_positive,
        negative_pairs,
        ensembl_to_uniprot,
        available_uniprot_ids,
        PDB_DIR
    )

    # Step 9: Balance dataset
    positive_samples = [d for d in dataset if d['interaction'] == 1]
    negative_samples = [d for d in dataset if d['interaction'] == 0]

    print(f"Dataset composition: {len(positive_samples):,} positive, {len(negative_samples):,} negative")

    # Balance to smaller class
    min_class_size = min(len(positive_samples), len(negative_samples))
    balanced_dataset = random.sample(positive_samples, min_class_size) + \
                       random.sample(negative_samples, min_class_size)
    random.shuffle(balanced_dataset)

    print(f"Balanced dataset: {len(balanced_dataset):,} total pairs")

    # Step 10: Save outputs
    save_dataset(balanced_dataset, OUTPUT_FILE)
    save_dataset(balanced_dataset, OUTPUT_FILE_SMALL, SMALL_SAMPLE_SIZE)

    # Step 11: Clean up temp file
    if os.path.exists(TEMP_HUMAN_FILE):
        print(f"Cleaning up temporary file: {TEMP_HUMAN_FILE}")
        os.remove(TEMP_HUMAN_FILE)

    print("=" * 80)
    print("Processing complete!")
    print(f"Full dataset: {OUTPUT_FILE}")
    print(f"Small dataset: {OUTPUT_FILE_SMALL}")
    print("=" * 80)


if __name__ == "__main__":
    main()
