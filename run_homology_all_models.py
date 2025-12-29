#!/usr/bin/env python
"""
Remote homology detection benchmark on ALL available models.

Tests all models in the registry with linear probe evaluation.
Uses 200 samples for quick comprehensive comparison.
"""
import sys
sys.path.insert(0, './mparsa/bench/procap-v2')

import json
import numpy as np
import pandas as pd
from pathlib import Path

from procap.models.registry import get_model
from procap.runners.classification import MulticlassClassificationRunner
from procap.tasks.schemas import TaskConfig, BloomLevel, TaskType


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Configuration
    data_path = Path("./mparsa/bench/procap-v2/data/homology/remote_homology.csv")
    output_path = Path("./mparsa/bench/procap-v2/benchmark_results/homology_all_models.json")
    num_samples = 200
    train_ratio = 0.8
    batch_size = 8
    seed = 42

    # ALL available models in registry
    models_to_test = [
        # ESM-2 family
        "esm2_t6_8M",
        "esm2_t12_35M",
        "esm2_t30_150M",
        "esm2_t33_650M",
        # ESM-1b
        "esm1b_t33_650M",
        # ProtBERT family
        "protbert",
        "protbert_bfd",
        # Other models
        "ontoprotein",
        "prott5_xl_bfd",
        "ankh_base",
        "prot_albert",
        "progen2_small",
        "protgpt2"
    ]

    print("=" * 80)
    print("Remote Homology Detection Benchmark - ALL MODELS")
    print("=" * 80)
    print(f"Data: {data_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Train ratio: {train_ratio}")
    print(f"Batch size: {batch_size}")
    print(f"Random seed: {seed}")
    print(f"Models to test ({len(models_to_test)}): {', '.join(models_to_test)}")
    print("=" * 80)
    print()

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Total samples in dataset: {len(df)}")
    print(f"Columns: {', '.join(df.columns.tolist())}")

    # Check unique classes
    if 'superfamily' in df.columns:
        unique_classes = df['superfamily'].nunique()
        print(f"Number of unique superfamilies: {unique_classes}")
    print()

    # Sample data for quicker run
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
        print(f"Sampled {num_samples} samples for benchmark")
    print()

    # Create task configuration
    task_config = TaskConfig(
        name="remote_homology",
        bloom_level=BloomLevel.ANALYSIS,
        domain="Evolutionary",
        description="Classify proteins into SCOP superfamilies based on structural homology",
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        dataset_path=str(data_path),
        input_fields=["sequence"],
        target_field="superfamily",
        metrics=["accuracy", "f1_macro"],
        batch_size=batch_size
    )

    # Store results for all models
    all_results = {}

    # Test each model
    for model_name in models_to_test:
        print("=" * 80)
        print(f"Testing model: {model_name}")
        print("=" * 80)

        try:
            # Load model
            print(f"Loading model {model_name}...")
            model = get_model(model_name)
            model.load()
            print(f"Model {model_name} loaded successfully")
            print(f"Model device: {model.device}")
            print()

            # Create runner
            runner = MulticlassClassificationRunner(
                model=model,
                task=task_config,
                batch_size=task_config.batch_size,
                show_progress=True,
                train_ratio=train_ratio
            )

            # Run evaluation
            print(f"Running evaluation for {model_name}...")
            results = runner.run(df)

            # Store results
            all_results[model_name] = results

            # Print results
            print()
            print(f"Results for {model_name}:")
            print("-" * 40)
            for metric_name, value in results.items():
                print(f"  {metric_name}: {value:.4f}")
            print()

            # Clean up
            del model
            del runner
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}
            print()

    # Save results
    print("=" * 80)
    print("Saving results...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY - Remote Homology Detection (All Models)")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1-Macro':<12} {'Status':<15}")
    print("-" * 80)

    for model_name, results in all_results.items():
        if "error" in results:
            print(f"{model_name:<20} {'N/A':<12} {'N/A':<12} {'FAILED':<15}")
        else:
            acc = results.get('accuracy', 0.0)
            f1 = results.get('f1_macro', 0.0)
            print(f"{model_name:<20} {acc:<12.4f} {f1:<12.4f} {'SUCCESS':<15}")

    print("=" * 80)

    # Print best performing models
    successful_models = {k: v for k, v in all_results.items() if "error" not in v}
    if successful_models:
        print()
        print("BEST PERFORMING MODELS:")
        print("-" * 80)

        # Sort by accuracy
        sorted_by_acc = sorted(successful_models.items(),
                               key=lambda x: x[1].get('accuracy', 0.0),
                               reverse=True)
        print("\nBy Accuracy:")
        for i, (model_name, results) in enumerate(sorted_by_acc[:5], 1):
            acc = results.get('accuracy', 0.0)
            f1 = results.get('f1_macro', 0.0)
            print(f"  {i}. {model_name}: Accuracy={acc:.4f}, F1-Macro={f1:.4f}")

        # Sort by F1-macro
        sorted_by_f1 = sorted(successful_models.items(),
                              key=lambda x: x[1].get('f1_macro', 0.0),
                              reverse=True)
        print("\nBy F1-Macro:")
        for i, (model_name, results) in enumerate(sorted_by_f1[:5], 1):
            acc = results.get('accuracy', 0.0)
            f1 = results.get('f1_macro', 0.0)
            print(f"  {i}. {model_name}: F1-Macro={f1:.4f}, Accuracy={acc:.4f}")

    print()
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
