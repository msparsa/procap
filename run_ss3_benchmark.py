#!/usr/bin/env python
import sys
sys.path.insert(0, './mparsa/bench/procap-v2')

import json
import numpy as np
import pandas as pd
from pathlib import Path

from procap.runners.token_classification import TokenClassificationRunner
from procap.tasks.schemas import TaskConfig, BloomLevel, TaskType
from procap.models.registry import get_model

# Load data
df = pd.read_csv('data/structure/secondary_structure_ss3.csv')
print(f"Loaded {len(df)} samples")

# Use subset for speed
df = df.head(500)

# Task config
task_config = TaskConfig(
    name='secondary_structure_ss3',
    task_type=TaskType.TOKEN_CLASSIFICATION,
    input_fields=['sequence'],
    target_field='ss3_labels',
    metrics=['accuracy', 'f1_macro'],
    batch_size=16,
    bloom_level=BloomLevel.UNDERSTANDING,
    domain='Structure',
    description='Predict 3-state secondary structure from sequence',
    dataset_path='data/structure/secondary_structure_ss3.csv',
)

results_dir = Path('benchmark_results')
results_dir.mkdir(exist_ok=True)

all_results = {}

for model_name in ['esm2_t6_8M', 'esm2_t33_650M', 'protbert']:
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print('='*50)

    np.random.seed(42)  # Reproducibility

    model = get_model(model_name)
    model.load()

    runner = TokenClassificationRunner(
        task=task_config,
        model=model,
        batch_size=16,
        show_progress=True,
        num_classes=3,
        train_ratio=0.8
    )

    results = runner.run(df)
    all_results[model_name] = results

    print(f"\nResults for {model_name}:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # Save individual result
    with open(results_dir / f'ss3_{model_name}.json', 'w') as f:
        json.dump({
            'model': model_name,
            'task': 'secondary_structure_ss3',
            'metrics': results
        }, f, indent=2)

    # Cleanup
    del model, runner
    import torch
    torch.cuda.empty_cache()

# Save combined results
with open(results_dir / 'ss3_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*50)
print("FINAL COMPARISON")
print("="*50)
print(f"{'Model':<20} {'Accuracy':>10} {'F1_macro':>10} {'H_acc':>8} {'E_acc':>8} {'C_acc':>8}")
print("-"*70)
for model, res in all_results.items():
    print(f"{model:<20} {res['accuracy']:>10.4f} {res['f1_macro']:>10.4f} {res['accuracy_H']:>8.4f} {res['accuracy_E']:>8.4f} {res['accuracy_C']:>8.4f}")
