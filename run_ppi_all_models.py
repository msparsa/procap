#!/usr/bin/env python3
"""
Comprehensive Protein-Protein Interaction Benchmark for All Available Models.

This script:
1. Loads PPI data from data/interaction/ppi_prediction.csv
2. Uses 200 samples for efficient benchmarking
3. Tests ALL available models from the registry
4. Uses PPIClassificationRunner from procap.runners.ppi
5. Saves results to benchmark_results/ppi_all_models.json
6. Configuration: train_ratio=0.8, batch_size=8, seed=42
7. Handles errors gracefully - continues testing even if some models fail
8. Cleans up GPU memory between models to prevent OOM errors
"""

import json
import os
import sys
from pathlib import Path
import gc

import numpy as np
import pandas as pd

# Try to import torch - if it fails, show helpful error message
try:
    import torch
except Exception as e:
    print("=" * 80)
    print("ERROR: PyTorch Import Failed")
    print("=" * 80)
    print(f"\nError: {e}")
    print("\nThis appears to be a PyTorch installation issue.")
    print("Please fix the PyTorch installation before running this benchmark.")
    print("\nSee RUN_PPI_BENCHMARK_INSTRUCTIONS.md for troubleshooting steps.")
    print("=" * 80)
    sys.exit(1)

# Add procap to path
sys.path.insert(0, str(Path(__file__).parent))

from procap.tasks.registry import get_task
from procap.runners.ppi import PPIClassificationRunner
from procap.models import get_model


def cleanup_gpu_memory():
    """Clean up GPU memory between models."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def main():
    """Run comprehensive PPI benchmark on all available models."""

    # Set random seed for reproducibility
    np.random.seed(42)

    print("=" * 80)
    print("Comprehensive Protein-Protein Interaction Benchmark")
    print("Testing ALL Available Models")
    print("=" * 80)

    # Configuration
    data_path = "data/interaction/ppi_prediction.csv"
    output_path = "benchmark_results/ppi_all_models.json"
    n_samples = 200
    train_ratio = 0.8
    batch_size = 8
    seed = 42

    # All models to test as specified by user
    # NOTE: progen2_small and protgpt2 are excluded due to compatibility issues:
    # - progen2_small: 'ProGenConfig' object has no attribute 'num_hidden_layers'
    # - protgpt2: CUDA error: device-side assert triggered (kills entire process)
    models_to_test = [
        # ESM-2 models
        "esm2_t6_8M",
        "esm2_t12_35M",
        "esm2_t30_150M",
        "esm2_t33_650M",
        # ESM-1b
        "esm1b_t33_650M",
        # ProtBERT models
        "protbert",
        "protbert_bfd",
        # OntoProtein
        "ontoprotein",
        # ProtT5
        "prott5_xl_bfd",
        # Ankh
        "ankh_base",
        # ProtAlbert
        "prot_albert",
    ]

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Total dataset size: {len(df)} samples")

    # Sample data
    print(f"Sampling {n_samples} samples with seed {seed}...")
    df_sample = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    # Check class balance
    n_positive = (df_sample['interaction'] == 1).sum()
    n_negative = (df_sample['interaction'] == 0).sum()
    print(f"Sample composition: {n_positive} positive, {n_negative} negative")

    # Get task configuration
    print("\nLoading task configuration...")
    task = get_task("ppi_prediction")
    print(f"Task: {task.name}")
    print(f"Task type: {task.task_type}")
    print(f"Metrics: {task.metrics}")

    # Track successful and failed models
    results = {}
    successful_models = []
    failed_models = []

    # Run benchmark for each model
    for i, model_name in enumerate(models_to_test, 1):
        print("\n" + "=" * 80)
        print(f"Testing Model {i}/{len(models_to_test)}: {model_name}")
        print("=" * 80)

        try:
            # Load model
            print(f"Loading model {model_name}...")
            model = get_model(model_name)
            model.load()
            print(f"Model {model_name} loaded successfully")

            # Create runner
            print("Creating PPIClassificationRunner...")
            runner = PPIClassificationRunner(
                model=model,
                task=task,
                batch_size=batch_size,
                combination="combined",  # Use combined embeddings strategy
                train_ratio=train_ratio,
                show_progress=True
            )

            # Run evaluation
            print(f"\nRunning evaluation on {n_samples} samples...")
            metrics = runner.run(df_sample)

            # Store results
            results[model_name] = metrics
            successful_models.append(model_name)

            print(f"\n{model_name} Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        except Exception as e:
            print(f"\nERROR testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {"error": str(e)}
            failed_models.append(model_name)

        finally:
            # Clean up GPU memory after each model
            print(f"\nCleaning up GPU memory after {model_name}...")
            cleanup_gpu_memory()

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    os.makedirs("benchmark_results", exist_ok=True)

    output_data = {
        "task": "ppi_prediction",
        "configuration": {
            "n_samples": n_samples,
            "train_ratio": train_ratio,
            "batch_size": batch_size,
            "seed": seed,
            "combination": "combined",
        },
        "data_statistics": {
            "n_positive": int(n_positive),
            "n_negative": int(n_negative),
        },
        "models_tested": models_to_test,
        "successful_models": successful_models,
        "failed_models": failed_models,
        "results": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\nTask: {task.name}")
    print(f"Samples: {n_samples} ({n_positive} positive, {n_negative} negative)")
    print(f"Train/Test Split: {train_ratio:.0%} / {(1-train_ratio):.0%}")
    print(f"Batch Size: {batch_size}")
    print(f"Seed: {seed}")

    print(f"\nModels Tested: {len(models_to_test)}")
    print(f"Successful: {len(successful_models)}")
    print(f"Failed: {len(failed_models)}")

    if failed_models:
        print(f"\nFailed Models:")
        for model_name in failed_models:
            error_msg = results[model_name].get("error", "Unknown error")
            print(f"  - {model_name}: {error_msg}")

    print(f"\nResults by Model:")
    print("-" * 80)

    for model_name in models_to_test:
        print(f"\n{model_name}:")
        if "error" in results[model_name]:
            print(f"  STATUS: FAILED")
            print(f"  ERROR: {results[model_name]['error']}")
        else:
            print(f"  STATUS: SUCCESS")
            for metric, value in results[model_name].items():
                print(f"  {metric}: {value:.4f}")

    # Print top performers if we have successful models
    if successful_models:
        print("\n" + "=" * 80)
        print("TOP PERFORMERS")
        print("=" * 80)

        # Rank by different metrics
        for metric in ["auroc", "auprc", "f1", "accuracy"]:
            # Get models that have this metric
            models_with_metric = [
                (name, results[name][metric])
                for name in successful_models
                if metric in results[name]
            ]

            if models_with_metric:
                # Sort by metric value (descending)
                ranked = sorted(models_with_metric, key=lambda x: x[1], reverse=True)

                print(f"\nTop 3 by {metric.upper()}:")
                for rank, (name, value) in enumerate(ranked[:3], 1):
                    print(f"  {rank}. {name}: {value:.4f}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
