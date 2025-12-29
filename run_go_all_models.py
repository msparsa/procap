#!/usr/bin/env python3
"""
GO term prediction benchmark with ALL available models in the registry.

Tests models:
- ESM2 family: esm2_t6_8M, esm2_t12_35M, esm2_t30_150M, esm2_t33_650M
- ESM1b: esm1b_t33_650M
- ProtBERT: protbert, protbert_bfd
- OntoProtein: ontoprotein
- ProtT5: prott5_xl_bfd, prott5_xl_uniref50
- Ankh: ankh_base, ankh_large
- ProtAlbert: prot_albert
- ProGen2: progen2_small, progen2_medium
- ProtGPT2: protgpt2

Uses 200 samples from gobench_remember.csv
Train ratio: 0.8 (80% training, 20% testing)
Batch size: 8 (to handle larger models)
"""

import sys
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

import gc
import json
import time
import numpy as np
import pandas as pd
import torch

from procap.models.registry import get_model
from procap.runners.classification import MultiLabelClassificationRunner
from procap.tasks.schemas import TaskConfig, BloomLevel, TaskType


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Configuration
    data_path = Path("./mparsa/bench/procap-v2/data/functional/gobench_remember.csv")
    output_path = Path("./mparsa/bench/procap-v2/benchmark_results/go_all_models.json")
    num_samples = 200
    train_ratio = 0.8
    batch_size = 8
    seed = 42

    # All models to test
    # NOTE: progen2_* and protgpt2 are excluded due to compatibility issues:
    # - progen2_*: 'ProGenConfig' object has no attribute 'num_hidden_layers'
    # - protgpt2: CUDA error: device-side assert triggered (kills entire process)
    models_to_test = [
        # ESM2 family
        "esm2_t6_8M",
        "esm2_t12_35M",
        "esm2_t30_150M",
        "esm2_t33_650M",
        # ESM1b
        "esm1b_t33_650M",
        # ProtBERT
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

    print("=" * 80)
    print("GO Term Prediction Benchmark - ALL MODELS")
    print("=" * 80)
    print(f"Data: {data_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Train ratio: {train_ratio}")
    print(f"Batch size: {batch_size}")
    print(f"Random seed: {seed}")
    print(f"Number of models to test: {len(models_to_test)}")
    print(f"Models: {', '.join(models_to_test)}")
    print("=" * 80)
    print()

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Total samples in dataset: {len(df)}")

    # Sample data for benchmark
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
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
        metrics=["f1_max", "auprc_micro"],
        batch_size=batch_size
    )

    # Store results for all models
    all_results = {}
    timing_info = {}

    # Test each model
    for i, model_name in enumerate(models_to_test):
        print("=" * 80)
        print(f"Testing model {i+1}/{len(models_to_test)}: {model_name}")
        print("=" * 80)

        start_time = time.time()

        try:
            # Load model
            print(f"Loading model {model_name}...")
            model = get_model(model_name)
            model.load()
            print(f"Model {model_name} loaded successfully")
            print(f"Model device: {model.device}")
            print()

            # Create runner with linear probe
            runner = MultiLabelClassificationRunner(
                model=model,
                task=task_config,
                batch_size=batch_size,
                show_progress=True,
                train_ratio=train_ratio
            )

            # Run evaluation
            print(f"Running evaluation for {model_name}...")
            results = runner.run(df)

            # Store results
            all_results[model_name] = results
            elapsed_time = time.time() - start_time
            timing_info[model_name] = {
                "elapsed_seconds": elapsed_time,
                "elapsed_minutes": elapsed_time / 60
            }

            # Print results
            print()
            print(f"Results for {model_name}:")
            print("-" * 40)
            for metric_name, value in results.items():
                print(f"  {metric_name}: {value:.4f}")
            print(f"  Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
            print()

            # Clean up GPU memory
            print("Cleaning up memory...")
            del model
            del runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print()

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}
            elapsed_time = time.time() - start_time
            timing_info[model_name] = {
                "elapsed_seconds": elapsed_time,
                "elapsed_minutes": elapsed_time / 60,
                "status": "failed"
            }
            print()

            # Clean up even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Save results
    print("=" * 80)
    print("Saving results...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine results and timing
    final_output = {
        "metadata": {
            "num_samples": num_samples,
            "train_ratio": train_ratio,
            "batch_size": batch_size,
            "seed": seed,
            "data_path": str(data_path),
            "models_tested": len(models_to_test),
            "models_succeeded": sum(1 for r in all_results.values() if "error" not in r),
            "models_failed": sum(1 for r in all_results.values() if "error" in r)
        },
        "results": all_results,
        "timing": timing_info
    }

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Results saved to {output_path}")
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful_models = []
    failed_models = []

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        if "error" in results:
            print(f"  ERROR: {results['error']}")
            failed_models.append(model_name)
        else:
            for metric_name, value in results.items():
                print(f"  {metric_name}: {value:.4f}")
            if model_name in timing_info:
                print(f"  time: {timing_info[model_name]['elapsed_minutes']:.2f} minutes")
            successful_models.append(model_name)

    print()
    print("=" * 80)
    print(f"Successful: {len(successful_models)}/{len(models_to_test)} models")
    if successful_models:
        print(f"  {', '.join(successful_models)}")
    print()
    print(f"Failed: {len(failed_models)}/{len(models_to_test)} models")
    if failed_models:
        print(f"  {', '.join(failed_models)}")
    print()

    # Print best performing models
    if successful_models:
        print("=" * 80)
        print("BEST PERFORMING MODELS")
        print("=" * 80)

        # Sort by f1_max
        f1_scores = {model: results.get("f1_max", 0.0)
                     for model, results in all_results.items()
                     if "error" not in results and "f1_max" in results}

        if f1_scores:
            sorted_by_f1 = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
            print("\nTop 5 by F1-Max:")
            for rank, (model, score) in enumerate(sorted_by_f1[:5], 1):
                print(f"  {rank}. {model}: {score:.4f}")

        # Sort by auprc_micro
        auprc_scores = {model: results.get("auprc_micro", 0.0)
                        for model, results in all_results.items()
                        if "error" not in results and "auprc_micro" in results}

        if auprc_scores:
            sorted_by_auprc = sorted(auprc_scores.items(), key=lambda x: x[1], reverse=True)
            print("\nTop 5 by AUPRC-Micro:")
            for rank, (model, score) in enumerate(sorted_by_auprc[:5], 1):
                print(f"  {rank}. {model}: {score:.4f}")
        print()

    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
