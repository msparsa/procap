#!/usr/bin/env python3
"""
Variant Effect Prediction Benchmark with Linear Probe

Evaluates protein language models on variant effect prediction using linear probe.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from procap.models.registry import get_model
from procap.runners.regression import RegressionRunner
from procap.tasks.schemas import TaskConfig, TaskType, BloomLevel


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Configuration
    data_path = Path("./mparsa/bench/procap-v2/data/variant_effect/proteingym_substitutions.csv")
    output_path = Path("./mparsa/bench/procap-v2/benchmark_results/variant_results.json")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Models to test
    model_names = ["esm2_t6_8M", "esm2_t33_650M", "protbert"]

    # Number of samples
    n_samples = 300

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Sample the data
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {n_samples} variants from dataset")

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())

    # Create task configuration for variant effect prediction
    task = TaskConfig(
        name="variant_effect_prediction",
        bloom_level=BloomLevel.ANALYSIS,
        domain="Functional Annotation",
        description="Predict the functional effect of sequence variants",
        task_type=TaskType.REGRESSION,
        dataset_path=str(data_path),
        input_fields=["mutated_sequence"],  # Use mutated sequence as input
        target_field="effect_score",
        metrics=["spearman", "pearson", "rmse", "mae", "r2"],
        compatible_models=model_names,
        batch_size=16
    )

    # Results dictionary
    results = {}

    # Evaluate each model
    for model_name in model_names:
        print("\n" + "="*80)
        print(f"Evaluating model: {model_name}")
        print("="*80)

        try:
            # Get model
            print(f"Loading model {model_name}...")
            model = get_model(model_name)
            model.load()

            # Create runner with linear probe
            runner = RegressionRunner(
                model=model,
                task=task,
                batch_size=16,
                show_progress=True,
                train_ratio=0.8  # 80% for training linear probe, 20% for testing
            )

            # Run evaluation
            print(f"\nRunning evaluation on {len(df)} samples...")
            metrics = runner.run(df)

            # Store results
            results[model_name] = metrics

            # Print results
            print(f"\nResults for {model_name}:")
            print(f"  Spearman correlation: {metrics.get('spearman', 'N/A'):.4f}")
            print(f"  Pearson correlation:  {metrics.get('pearson', 'N/A'):.4f}")
            print(f"  RMSE:                 {metrics.get('rmse', 'N/A'):.4f}")
            print(f"  MAE:                  {metrics.get('mae', 'N/A'):.4f}")
            print(f"  R²:                   {metrics.get('r2', 'N/A'):.4f}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {"error": str(e)}

    # Save results
    print(f"\n" + "="*80)
    print("Saving results...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Dataset: {data_path}")
    print(f"Samples: {n_samples}")
    print(f"Models tested: {len(model_names)}")
    print("\nSpearman Correlation Results:")
    print("-" * 40)
    for model_name in model_names:
        if model_name in results and "spearman" in results[model_name]:
            spearman = results[model_name]["spearman"]
            print(f"  {model_name:20s}: {spearman:.4f}")
        else:
            print(f"  {model_name:20s}: FAILED")
    print("="*80)


if __name__ == "__main__":
    main()
