#!/usr/bin/env python3
"""
Comprehensive protein family classification benchmark across ALL available models.

Tests models: esm2_t6_8M, esm2_t12_35M, esm2_t30_150M, esm2_t33_650M,
              esm1b_t33_650M, protbert, protbert_bfd, ontoprotein,
              prott5_xl_bfd, ankh_base, prot_albert, progen2_small, protgpt2

Uses 200 samples for evaluation with linear probe.
"""

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
    data_path = Path("./mparsa/bench/procap-v2/data/functional/family_classification.csv")
    output_path = Path("./mparsa/bench/procap-v2/benchmark_results/family_all_models.json")
    num_samples = None  # Use full dataset (set to integer to limit samples)

    # All available models from the registry
    models_to_test = [
        "esm2_t6_8M",
        "esm2_t12_35M",
        "esm2_t30_150M",
        "esm2_t33_650M",
        "esm1b_t33_650M",
        "protbert",
        "protbert_bfd",
        "ontoprotein",
        "prott5_xl_bfd",
        "ankh_base",
        "prot_albert",
        "progen2_small",
        "protgpt2"
    ]

    print("=" * 80)
    print("Protein Family Classification Benchmark - ALL MODELS")
    print("=" * 80)
    print(f"Data: {data_path}")
    print(f"Number of samples: {'Full dataset' if num_samples is None else num_samples}")
    print(f"Number of models to test: {len(models_to_test)}")
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Random seed: 42")
    print(f"Train ratio: 0.8")
    print(f"Batch size: 8")
    print("=" * 80)
    print()

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Total samples in dataset: {len(df)}")

    # Filter out UNKNOWN family
    df = df[df['family'] != 'UNKNOWN'].reset_index(drop=True)
    print(f"Samples after removing UNKNOWN family: {len(df)}")

    # Get family counts
    family_counts = df['family'].value_counts()
    print(f"Number of unique families: {len(family_counts)}")
    print(f"Family distribution (top 10):")
    for family, count in family_counts.head(10).items():
        print(f"  {family}: {count}")
    print()

    # Sample data for quicker run (or use full dataset if num_samples is None)
    if num_samples is not None and len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {num_samples} samples for benchmark")
    else:
        print(f"Using full dataset: {len(df)} samples for benchmark")
    print()

    # Create task configuration
    task_config = TaskConfig(
        name="protein_family_classification",
        bloom_level=BloomLevel.REMEMBERING,
        domain="Functional Annotation",
        description="Classify proteins into Pfam families from sequence",
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        dataset_path=str(data_path),
        input_fields=["sequence"],
        target_field="family",
        metrics=["accuracy", "f1_macro"],
        batch_size=8
    )

    # Store results for all models
    all_results = {}
    successful_models = []
    failed_models = []

    # Test each model
    for i, model_name in enumerate(models_to_test, 1):
        print("=" * 80)
        print(f"Testing model {i}/{len(models_to_test)}: {model_name}")
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
                train_ratio=0.8
            )

            # Run evaluation
            print(f"Running evaluation for {model_name}...")
            results = runner.run(df)

            # Store results
            all_results[model_name] = results
            successful_models.append(model_name)

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
            failed_models.append(model_name)
            print()

            # Try to clean up on error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

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
    print("SUMMARY")
    print("=" * 80)
    print(f"\nSuccessful models: {len(successful_models)}/{len(models_to_test)}")
    print(f"Failed models: {len(failed_models)}/{len(models_to_test)}")
    print()

    if successful_models:
        print("Successful models results:")
        print("-" * 80)
        for model_name in successful_models:
            results = all_results[model_name]
            print(f"\n{model_name}:")
            for metric_name, value in results.items():
                print(f"  {metric_name}: {value:.4f}")

    if failed_models:
        print()
        print("Failed models:")
        print("-" * 80)
        for model_name in failed_models:
            results = all_results[model_name]
            print(f"\n{model_name}:")
            print(f"  ERROR: {results['error']}")

    print()
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
