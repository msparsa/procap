#!/usr/bin/env python3
"""
GO term prediction benchmark with linear probe evaluation.

Tests models: esm2_t6_8M, esm2_t33_650M, protbert
Uses 300 samples for quicker run.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from procap.models.registry import get_model
from procap.runners.classification import MultiLabelClassificationRunner
from procap.tasks.schemas import TaskConfig, BloomLevel, TaskType


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Configuration
    data_path = Path("./mparsa/bench/procap-v2/data/functional/gobench_remember.csv")
    output_path = Path("./mparsa/bench/procap-v2/benchmark_results/go_results.json")
    num_samples = 300
    models_to_test = ["esm2_t6_8M", "esm2_t33_650M", "protbert"]

    print("=" * 80)
    print("GO Term Prediction Benchmark - Linear Probe Evaluation")
    print("=" * 80)
    print(f"Data: {data_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Random seed: 42")
    print("=" * 80)
    print()

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Total samples in dataset: {len(df)}")

    # Sample data for quicker run
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {num_samples} samples for benchmark")
    print()

    # Create task configuration
    task_config = TaskConfig(
        name="go_term_prediction",
        bloom_level=BloomLevel.REMEMBERING,
        domain="Functional Annotation",
        description="Predict GO terms for proteins from sequence",
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
        dataset_path=str(data_path),
        input_fields=["sequence"],
        target_field="go_terms",
        metrics=["f1_max", "auprc_micro", "auroc_micro"],
        batch_size=16
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
            runner = MultiLabelClassificationRunner(
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
    print("SUMMARY")
    print("=" * 80)
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        if "error" in results:
            print(f"  ERROR: {results['error']}")
        else:
            for metric_name, value in results.items():
                print(f"  {metric_name}: {value:.4f}")
    print()
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
