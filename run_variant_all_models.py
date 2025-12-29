#!/usr/bin/env python3
"""
Comprehensive Variant Effect Prediction Benchmark - ALL Models

Evaluates ALL available protein language models on variant effect prediction
using Ridge regression. Tests 14 different models including ESM2, ESM1b, ESM1v,
ProtBERT, ProtT5, OntoProtein, ANKH, ProtALBERT, ProGen2, and ProtGPT2.
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
    output_path = Path("./mparsa/bench/procap-v2/benchmark_results/variant_all_models.json")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ALL available models from the registry
    model_names = [
        # ESM2 family (4 models)
        "esm2_t6_8M",
        "esm2_t12_35M",
        "esm2_t30_150M",
        "esm2_t33_650M",
        # ESM1 family (3 models)
        "esm1b_t33_650M",
        "esm1v_t33_650M_1",  # Variant-specific model
        # ProtBERT family (2 models)
        "protbert",
        "protbert_bfd",
        # Other models (5 models)
        "ontoprotein",
        "prott5_xl_bfd",
        "ankh_base",
        "prot_albert",
        "progen2_small",
        "protgpt2",
    ]

    # Number of samples
    n_samples = None  # Use full dataset (set to integer to limit samples)

    # Hyperparameters
    train_ratio = 0.8
    batch_size = 8
    seed = 42

    print("="*80)
    print("COMPREHENSIVE VARIANT EFFECT PREDICTION BENCHMARK")
    print("="*80)
    print(f"Models to test: {len(model_names)}")
    print(f"Samples: {'Full dataset' if n_samples is None else n_samples}")
    print(f"Train ratio: {train_ratio}")
    print(f"Batch size: {batch_size}")
    print(f"Random seed: {seed}")
    print("="*80)

    print(f"\nLoading data from {data_path}")
    df = pd.read_csv(data_path)

    # Sample the data (or use full dataset if n_samples is None)
    if n_samples is not None and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
        print(f"Sampled {n_samples} variants from dataset")
    else:
        print(f"Using full dataset: {len(df)} variants for benchmark")

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head(3))

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
        batch_size=batch_size
    )

    # Results dictionary
    results = {}
    successful_models = []
    failed_models = []

    # Evaluate each model
    for i, model_name in enumerate(model_names, 1):
        print("\n" + "="*80)
        print(f"[{i}/{len(model_names)}] Evaluating model: {model_name}")
        print("="*80)

        try:
            # Get model
            print(f"Loading model {model_name}...")
            model = get_model(model_name)
            model.load()

            # Create runner with linear probe (Ridge regression)
            runner = RegressionRunner(
                model=model,
                task=task,
                batch_size=batch_size,
                show_progress=True,
                train_ratio=train_ratio
            )

            # Run evaluation
            print(f"\nRunning evaluation on {len(df)} samples...")
            metrics = runner.run(df)

            # Store results
            results[model_name] = metrics
            successful_models.append(model_name)

            # Print results
            print(f"\nResults for {model_name}:")
            print(f"  Spearman correlation: {metrics.get('spearman', 'N/A'):.4f}")
            print(f"  Pearson correlation:  {metrics.get('pearson', 'N/A'):.4f}")
            print(f"  RMSE:                 {metrics.get('rmse', 'N/A'):.4f}")
            print(f"  MAE:                  {metrics.get('mae', 'N/A'):.4f}")
            print(f"  R²:                   {metrics.get('r2', 'N/A'):.4f}")

        except Exception as e:
            print(f"\n!!! Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {"error": str(e)}
            failed_models.append(model_name)
            print(f"Continuing to next model...")

    # Save results
    print(f"\n" + "="*80)
    print("Saving results...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    # Print comprehensive summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Dataset: {data_path}")
    print(f"Samples: {len(df)}")
    print(f"Train ratio: {train_ratio}")
    print(f"Batch size: {batch_size}")
    print(f"Random seed: {seed}")
    print(f"\nModels tested: {len(model_names)}")
    print(f"Successful: {len(successful_models)}")
    print(f"Failed: {len(failed_models)}")

    if successful_models:
        print("\n" + "="*80)
        print("SPEARMAN CORRELATION RESULTS (PRIMARY METRIC)")
        print("="*80)

        # Sort by Spearman correlation
        spearman_results = []
        for model_name in successful_models:
            if "spearman" in results[model_name]:
                spearman_results.append((model_name, results[model_name]["spearman"]))

        spearman_results.sort(key=lambda x: x[1], reverse=True)

        print(f"{'Rank':<6} {'Model':<25} {'Spearman':<12} {'Pearson':<12} {'R²':<12}")
        print("-" * 80)

        for rank, (model_name, spearman) in enumerate(spearman_results, 1):
            pearson = results[model_name].get("pearson", float('nan'))
            r2 = results[model_name].get("r2", float('nan'))
            print(f"{rank:<6} {model_name:<25} {spearman:>11.4f} {pearson:>11.4f} {r2:>11.4f}")

        print("\n" + "="*80)
        print("ALL METRICS FOR SUCCESSFUL MODELS")
        print("="*80)
        for model_name in successful_models:
            print(f"\n{model_name}:")
            for metric, value in results[model_name].items():
                if metric != "error":
                    print(f"  {metric:<20}: {value:.4f}")

    if failed_models:
        print("\n" + "="*80)
        print("FAILED MODELS")
        print("="*80)
        for model_name in failed_models:
            error_msg = results[model_name].get("error", "Unknown error")
            print(f"  {model_name:<25}: {error_msg}")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
