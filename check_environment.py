#!/usr/bin/env python3
"""
Environment checker script for ProCap benchmark.
Run this to verify your environment is set up correctly.
"""

import sys
print("=" * 70)
print("PROCAP ENVIRONMENT CHECKER")
print("=" * 70)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[0]}")
print()

# Check for required packages
packages_to_check = [
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "sklearn",
    "tqdm",
    "pydantic",
]

print("Checking required packages:")
print("-" * 70)

missing_packages = []
for package in packages_to_check:
    try:
        mod = __import__(package)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {package:20s} {version}")
    except ImportError:
        print(f"✗ {package:20s} NOT FOUND")
        missing_packages.append(package)

print()

# Check for procap
print("Checking procap package:")
print("-" * 70)
try:
    sys.path.insert(0, "./mparsa/bench/procap-v2")
    import procap
    print(f"✓ procap found at: {procap.__file__}")

    # Try importing key components
    try:
        from procap.models.registry import get_model, list_models
        print(f"✓ procap.models.registry imported successfully")
        models = list_models()
        print(f"  Available models: {len(models)}")
    except Exception as e:
        print(f"✗ Error importing procap.models.registry: {e}")

except ImportError as e:
    print(f"✗ procap NOT FOUND: {e}")

print()

if missing_packages:
    print("=" * 70)
    print("MISSING PACKAGES:")
    print("=" * 70)
    print(f"The following packages are missing: {', '.join(missing_packages)}")
    print()
    print("To install, run:")
    print("  pip install " + " ".join(missing_packages))
    print()
    print("Or install the full procap package:")
    print("  pip install -e ./mparsa/bench/procap-v2")
    print("=" * 70)
    sys.exit(1)
else:
    print("=" * 70)
    print("✓ ALL REQUIRED PACKAGES FOUND!")
    print("=" * 70)
    print()
    print("Your environment is ready to run the benchmark.")
    print("Run the following command:")
    print()
    print("  cd ./mparsa/bench/procap-v2")
    print(f"  {sys.executable} run_go_all_models.py")
    print()
    sys.exit(0)
