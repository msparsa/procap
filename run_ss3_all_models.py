#!/usr/bin/env python
"""
SS3 Secondary Structure Benchmark - All Models
Runs token classification benchmark on all available models with 200 samples each.
"""
import sys
sys.path.insert(0, './mparsa/bench/procap-v2')

import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import gc

from procap.runners.token_classification import TokenClassificationRunner
from procap.tasks.schemas import TaskConfig, BloomLevel, TaskType
from procap.models.registry import get_model

# Configuration
NUM_SAMPLES = None  # Use full dataset (set to integer to limit samples)
SEED = 42
TRAIN_RATIO = 0.8
BATCH_SIZE = 16

# Models to test
MODELS_TO_TEST = [
    'esm2_t6_8M',
    'esm2_t12_35M',
    'esm2_t33_650M',
    'protbert',
    'protbert_bfd',
    'ankh_base',
]

print("="*70)
print("SS3 SECONDARY STRUCTURE BENCHMARK - ALL MODELS")
print("="*70)
print(f"Configuration:")
print(f"  Samples: {'Full dataset' if NUM_SAMPLES is None else NUM_SAMPLES}")
print(f"  Train ratio: {TRAIN_RATIO}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Random seed: {SEED}")
print(f"  Models: {', '.join(MODELS_TO_TEST)}")
print("="*70)

# Load data
data_path = './mparsa/bench/procap-v2/data/structure/secondary_structure_ss3.csv'
df = pd.read_csv(data_path)
print(f"\nLoaded {len(df)} total samples from dataset")

# Use specified number of samples (or full dataset if NUM_SAMPLES is None)
if NUM_SAMPLES is not None:
    df = df.head(NUM_SAMPLES)
    print(f"Using {NUM_SAMPLES} samples for benchmark")
else:
    print(f"Using full dataset: {len(df)} samples for benchmark")

# Task config
task_config = TaskConfig(
    name='secondary_structure_ss3',
    task_type=TaskType.TOKEN_CLASSIFICATION,
    input_fields=['sequence'],
    target_field='ss3_labels',
    metrics=['accuracy', 'f1_macro'],
    batch_size=BATCH_SIZE,
    bloom_level=BloomLevel.UNDERSTANDING,
    domain='Structure',
    description='Predict 3-state secondary structure from sequence',
    dataset_path='data/structure/secondary_structure_ss3.csv',
)

# Results storage
results_dir = Path('./mparsa/bench/procap-v2/benchmark_results')
results_dir.mkdir(exist_ok=True)

all_results = {}
failed_models = []

# Run benchmark for each model
for model_name in MODELS_TO_TEST:
    print(f"\n{'='*70}")
    print(f"TESTING: {model_name}")
    print('='*70)

    try:
        # Set seed for reproducibility
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # Load model
        print(f"Loading model: {model_name}")
        model = get_model(model_name)
        model.load()
        print(f"Model loaded successfully")

        # Create runner
        runner = TokenClassificationRunner(
            task=task_config,
            model=model,
            batch_size=BATCH_SIZE,
            show_progress=True,
            num_classes=3,
            train_ratio=TRAIN_RATIO
        )

        # Run benchmark
        print(f"Running benchmark...")
        results = runner.run(df)
        all_results[model_name] = results

        print(f"\nResults for {model_name}:")
        print(f"  Overall Accuracy: {results['accuracy']:.4f}")
        print(f"  F1 Macro: {results['f1_macro']:.4f}")
        print(f"  Per-class accuracy:")
        print(f"    Helix (H): {results.get('accuracy_H', 0):.4f}")
        print(f"    Sheet (E): {results.get('accuracy_E', 0):.4f}")
        print(f"    Coil  (C): {results.get('accuracy_C', 0):.4f}")

        # Save individual result
        individual_result = {
            'model': model_name,
            'task': 'secondary_structure_ss3',
            'num_samples': NUM_SAMPLES,
            'train_ratio': TRAIN_RATIO,
            'seed': SEED,
            'metrics': results
        }

        with open(results_dir / f'ss3_{model_name}.json', 'w') as f:
            json.dump(individual_result, f, indent=2)

        print(f"✓ {model_name} completed successfully")

    except Exception as e:
        print(f"\n✗ ERROR testing {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        failed_models.append(model_name)
        all_results[model_name] = {'error': str(e)}

    finally:
        # Cleanup GPU memory
        print(f"\nCleaning up GPU memory...")
        try:
            del model
            del runner
        except:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"Cleanup complete\n")

# Save combined results
combined_results = {
    'config': {
        'num_samples': NUM_SAMPLES,
        'train_ratio': TRAIN_RATIO,
        'batch_size': BATCH_SIZE,
        'seed': SEED,
        'models_tested': MODELS_TO_TEST,
        'failed_models': failed_models
    },
    'results': all_results
}

output_file = results_dir / 'ss3_all_models.json'
with open(output_file, 'w') as f:
    json.dump(combined_results, f, indent=2)

print(f"\n{'='*70}")
print("FINAL RESULTS - SS3 BENCHMARK COMPARISON")
print('='*70)

# Print comparison table
if all_results:
    print(f"\n{'Model':<20} {'Accuracy':>10} {'F1_macro':>10} {'H_acc':>8} {'E_acc':>8} {'C_acc':>8}")
    print("-"*70)

    successful_models = {k: v for k, v in all_results.items() if 'error' not in v}

    for model_name in MODELS_TO_TEST:
        if model_name in successful_models:
            res = all_results[model_name]
            print(f"{model_name:<20} "
                  f"{res['accuracy']:>10.4f} "
                  f"{res['f1_macro']:>10.4f} "
                  f"{res.get('accuracy_H', 0):>8.4f} "
                  f"{res.get('accuracy_E', 0):>8.4f} "
                  f"{res.get('accuracy_C', 0):>8.4f}")
        else:
            print(f"{model_name:<20} {'FAILED':>10}")

if failed_models:
    print(f"\n⚠ Failed models: {', '.join(failed_models)}")

print(f"\n{'='*70}")
print(f"Results saved to: {output_file}")
print(f"Individual results saved to: {results_dir}/ss3_<model_name>.json")
print('='*70)
