#!/usr/bin/env python3
"""
Download external data files for ProCap-v2 benchmarks.

This script downloads large data files that are not included in the repository
due to size constraints. Currently downloads:
- ProteinGym DMS substitutions (~2GB)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --data-dir /custom/path
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional


# Data sources configuration
DATA_SOURCES = {
    "proteingym_substitutions": {
        "url": "https://marks.hms.harvard.edu/proteingym/DMS_substitutions.csv",
        "filename": "proteingym_substitutions.csv",
        "target_dir": "data/variant_effect",
        "size_mb": 2000,
        "description": "ProteinGym DMS substitutions (Deep Mutational Scanning data)",
    },
}


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB."""
    return filepath.stat().st_size / (1024 * 1024)


def download_with_progress(url: str, filepath: Path, expected_size_mb: float) -> bool:
    """Download a file with progress indicator."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {filepath}")
    print(f"Expected size: ~{expected_size_mb:.0f} MB")
    print()

    try:
        # Create parent directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(
                    f"\rProgress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)"
                )
            else:
                downloaded_mb = downloaded / (1024 * 1024)
                sys.stdout.write(f"\rDownloaded: {downloaded_mb:.1f} MB")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, filepath, reporthook)
        print("\n")
        return True

    except Exception as e:
        print(f"\nError downloading file: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def verify_file(filepath: Path, min_size_mb: float = 100) -> bool:
    """Verify downloaded file is valid."""
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return False

    size_mb = get_file_size_mb(filepath)
    if size_mb < min_size_mb:
        print(f"File too small: {size_mb:.1f} MB (expected >{min_size_mb} MB)")
        return False

    # Check if it's a valid CSV (has header line)
    try:
        with open(filepath, "r") as f:
            header = f.readline()
            if not header or "," not in header:
                print("File doesn't appear to be a valid CSV")
                return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    print(f"File verified: {size_mb:.1f} MB")
    return True


def download_proteingym(data_dir: Path) -> bool:
    """Download ProteinGym substitutions data."""
    source = DATA_SOURCES["proteingym_substitutions"]
    target_path = data_dir / source["target_dir"] / source["filename"]

    print("=" * 60)
    print(f"Downloading: {source['description']}")
    print("=" * 60)

    # Check if already exists
    if target_path.exists():
        size_mb = get_file_size_mb(target_path)
        if size_mb > 100:  # Reasonable minimum size
            print(f"File already exists: {target_path} ({size_mb:.1f} MB)")
            response = input("Re-download? [y/N]: ").strip().lower()
            if response != "y":
                print("Skipping download.")
                return True

    # Download
    success = download_with_progress(
        source["url"], target_path, source["size_mb"]
    )

    if success:
        if verify_file(target_path):
            print(f"Successfully downloaded: {target_path}")
            return True
        else:
            print("Download verification failed!")
            return False
    else:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download external data files for ProCap-v2"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base directory for data (default: repository root)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available downloads without downloading",
    )
    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Use repository root (parent of scripts/)
        data_dir = Path(__file__).parent.parent

    if args.list:
        print("Available downloads:")
        print("-" * 40)
        for name, info in DATA_SOURCES.items():
            target = data_dir / info["target_dir"] / info["filename"]
            exists = "EXISTS" if target.exists() else "NOT FOUND"
            print(f"  {name}:")
            print(f"    Description: {info['description']}")
            print(f"    Size: ~{info['size_mb']} MB")
            print(f"    Target: {target}")
            print(f"    Status: {exists}")
            print()
        return

    print("ProCap-v2 Data Downloader")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print()

    # Download ProteinGym data
    success = download_proteingym(data_dir)

    print()
    print("=" * 60)
    if success:
        print("Download complete!")
        print()
        print("You can now run variant effect benchmarks with the full dataset:")
        print("  python run_variant_all_models.py")
    else:
        print("Download failed. Please try again or download manually from:")
        print(f"  {DATA_SOURCES['proteingym_substitutions']['url']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
