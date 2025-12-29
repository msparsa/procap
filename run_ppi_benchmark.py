#!/usr/bin/env python3
"""
Protein-Protein Interaction Benchmark with Linear Probe.

This script:
1. Loads PPI data from data/interaction/ppi_prediction.csv
2. Uses 300 samples for quicker run
3. Tests models: esm2_t6_8M, esm2_t33_650M, protbert
4. Uses PPIClassificationRunner from procap.runners.ppi
5. Sets np.random.seed(42) for reproducibility
6. Saves results to benchmark_results/ppi_results.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add procap to path
sys.path.insert(0, str(Path(__file__).parent))

from procap.tasks.registry import get_task
from procap.runners.ppi import PPIClassificationRunner
from procap.models import get_model


def main():
    """Run PPI benchmark with linear probe."""

    # Set random seed for reproducibility
    np.random.seed(42)

    print("=" * 80)
    print("Protein-Protein Interaction Benchmark - Linear Probe Evaluation")
    print("=" * 80)

    # Configuration
    data_path = "data/interaction/ppi_prediction.csv"
    output_path = "benchmark_results/ppi_results.json"
    n_samples = 200
    models_to_test = ["esm2_t6_8M", "esm2_t33_650M", "protbert"]

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Total dataset size: {len(df)} samples")

    # Sample data for quicker run
    print(f"Sampling {n_samples} samples...")
    df_sample = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

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

    # Run benchmark for each model
    results = {}

    for model_name in models_to_test:
        print("\n" + "=" * 80)
        print(f"Testing Model: {model_name}")
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
                batch_size=8,
                combination="combined",  # Use combined embeddings strategy
                train_ratio=0.8,  # 80% train, 20% test
                show_progress=True
            )

            # Run evaluation
            print(f"\nRunning evaluation on {n_samples} samples...")
            metrics = runner.run(df_sample)

            # Store results
            results[model_name] = metrics

            print(f"\n{model_name} Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        except Exception as e:
            print(f"\nError testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {"error": str(e)}

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    os.makedirs("benchmark_results", exist_ok=True)

    output_data = {
        "task": "ppi_prediction",
        "n_samples": n_samples,
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "models": models_to_test,
        "results": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(f"\nTask: {task.name}")
    print(f"Samples: {n_samples} ({n_positive} positive, {n_negative} negative)")
    print(f"\nResults:")

    for model_name in models_to_test:
        print(f"\n{model_name}:")
        if "error" in results[model_name]:
            print(f"  ERROR: {results[model_name]['error']}")
        else:
            for metric, value in results[model_name].items():
                print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
