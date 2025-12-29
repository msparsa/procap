#!/usr/bin/env python3
"""
Create comprehensive visualizations for ProCap benchmark results.

Generates publication-ready figures for all benchmark tasks.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
})

# Model display names and colors
MODEL_DISPLAY_NAMES = {
    'esm2_t6_8M': 'ESM2-8M',
    'esm2_t12_35M': 'ESM2-35M',
    'esm2_t33_650M': 'ESM2-650M',
    'protbert': 'ProtBERT',
    'protbert_bfd': 'ProtBERT-BFD',
    'ankh_base': 'Ankh-Base',
}

MODEL_COLORS = {
    'esm2_t6_8M': '#1f77b4',
    'esm2_t12_35M': '#2ca02c',
    'esm2_t33_650M': '#ff7f0e',
    'protbert': '#d62728',
    'protbert_bfd': '#9467bd',
    'ankh_base': '#8c564b',
}

MODEL_SIZES = {
    'esm2_t6_8M': 8,
    'esm2_t12_35M': 35,
    'esm2_t33_650M': 650,
    'protbert': 420,
    'protbert_bfd': 420,
    'ankh_base': 350,
}

# Task display names
TASK_DISPLAY_NAMES = {
    'ss3': 'SS3 Prediction',
    'go_multilabel': 'GO Term Prediction',
    'ppi': 'PPI Prediction',
    'variant': 'Variant Effect',
    'family': 'Family Classification',
    'homology': 'Remote Homology',
}


def load_all_results(results_dir: Path) -> Dict[str, Any]:
    """Load all benchmark results from JSON files."""
    results = {}

    # SS3 results
    ss3_path = results_dir / 'ss3_all_models.json'
    if ss3_path.exists():
        with open(ss3_path) as f:
            data = json.load(f)
            results['ss3'] = data.get('results', data)

    # GO multilabel results
    go_path = results_dir / 'go_multilabel_results.json'
    if go_path.exists():
        with open(go_path) as f:
            results['go_multilabel'] = json.load(f)

    # PPI results
    ppi_path = results_dir / 'ppi_results.json'
    if ppi_path.exists():
        with open(ppi_path) as f:
            data = json.load(f)
            results['ppi'] = data.get('results', data)

    # Variant results
    variant_path = results_dir / 'variant_results.json'
    if variant_path.exists():
        with open(variant_path) as f:
            results['variant'] = json.load(f)

    # Family results
    family_path = results_dir / 'family_results.json'
    if family_path.exists():
        with open(family_path) as f:
            results['family'] = json.load(f)

    # Homology results
    homology_path = results_dir / 'homology_results.json'
    if homology_path.exists():
        with open(homology_path) as f:
            results['homology'] = json.load(f)

    return results


def create_ss3_comparison_chart(results: Dict, output_path: Path):
    """Create bar chart comparing SS3 performance across models."""
    if 'ss3' not in results:
        print("No SS3 results found")
        return

    ss3_data = results['ss3']
    models = []
    accuracies = []

    for model, metrics in ss3_data.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            models.append(MODEL_DISPLAY_NAMES.get(model, model))
            accuracies.append(metrics['accuracy'])

    # Sort by accuracy
    sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
    models, accuracies = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [MODEL_COLORS.get(m.lower().replace('-', '_').replace('esm2', 'esm2_t'), '#1f77b4')
              for m in models]

    # Create color list based on display names
    color_list = []
    for m in models:
        for key, display in MODEL_DISPLAY_NAMES.items():
            if display == m:
                color_list.append(MODEL_COLORS.get(key, '#1f77b4'))
                break
        else:
            color_list.append('#1f77b4')

    bars = ax.bar(models, accuracies, color=color_list, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Secondary Structure Prediction (SS3) - Model Comparison')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_ss3_per_class_chart(results: Dict, output_path: Path):
    """Create grouped bar chart for SS3 per-class accuracy."""
    if 'ss3' not in results:
        return

    ss3_data = results['ss3']
    models = []
    h_acc = []
    e_acc = []
    c_acc = []

    for model, metrics in ss3_data.items():
        if isinstance(metrics, dict) and 'accuracy_H' in metrics:
            models.append(MODEL_DISPLAY_NAMES.get(model, model))
            h_acc.append(metrics['accuracy_H'])
            e_acc.append(metrics['accuracy_E'])
            c_acc.append(metrics['accuracy_C'])

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_h = ax.bar(x - width, h_acc, width, label='Helix (H)', color='#E53935')
    bars_e = ax.bar(x, e_acc, width, label='Strand (E)', color='#1E88E5')
    bars_c = ax.bar(x + width, c_acc, width, label='Coil (C)', color='#43A047')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Secondary Structure Prediction: Per-Class Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bars in [bars_h, bars_e, bars_c]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only show labels for non-trivial values
                ax.annotate(f'{height:.0%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_task_heatmap(results: Dict, output_path: Path):
    """Create heatmap of model performance across all tasks."""
    # Collect metrics for each task-model pair
    data = {}

    # SS3 - accuracy
    if 'ss3' in results:
        for model, metrics in results['ss3'].items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                display_name = MODEL_DISPLAY_NAMES.get(model, model)
                if display_name not in data:
                    data[display_name] = {}
                data[display_name]['SS3'] = metrics['accuracy']

    # GO - f1_max
    if 'go_multilabel' in results:
        for model, metrics in results['go_multilabel'].items():
            if isinstance(metrics, dict) and 'f1_max' in metrics:
                display_name = MODEL_DISPLAY_NAMES.get(model, model)
                if display_name not in data:
                    data[display_name] = {}
                data[display_name]['GO Terms'] = metrics['f1_max']

    # PPI - auroc
    if 'ppi' in results:
        for model, metrics in results['ppi'].items():
            if isinstance(metrics, dict) and 'auroc' in metrics:
                display_name = MODEL_DISPLAY_NAMES.get(model, model)
                if display_name not in data:
                    data[display_name] = {}
                data[display_name]['PPI'] = metrics['auroc']

    # Variant - spearman
    if 'variant' in results:
        for model, metrics in results['variant'].items():
            if isinstance(metrics, dict) and 'spearman' in metrics:
                display_name = MODEL_DISPLAY_NAMES.get(model, model)
                if display_name not in data:
                    data[display_name] = {}
                data[display_name]['Variant Effect'] = metrics['spearman']

    # Family - accuracy
    if 'family' in results:
        for model, metrics in results['family'].items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                display_name = MODEL_DISPLAY_NAMES.get(model, model)
                if display_name not in data:
                    data[display_name] = {}
                data[display_name]['Family'] = metrics['accuracy']

    # Homology - accuracy
    if 'homology' in results:
        for model, metrics in results['homology'].items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                display_name = MODEL_DISPLAY_NAMES.get(model, model)
                if display_name not in data:
                    data[display_name] = {}
                data[display_name]['Homology'] = metrics['accuracy']

    # Convert to DataFrame
    df = pd.DataFrame(data).T

    # Reorder columns
    column_order = ['SS3', 'GO Terms', 'PPI', 'Variant Effect', 'Family', 'Homology']
    df = df[[c for c in column_order if c in df.columns]]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': 'Performance Score'}
    )

    ax.set_title('Model Performance Across All Tasks\n(Accuracy/AUROC/Spearman/F1-max)')
    ax.set_xlabel('Task')
    ax.set_ylabel('Model')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_model_size_vs_performance(results: Dict, output_path: Path):
    """Create scatter plot of model size vs SS3 performance."""
    if 'ss3' not in results:
        return

    models = []
    sizes = []
    scores = []

    for model, metrics in results['ss3'].items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            if model in MODEL_SIZES:
                models.append(model)
                sizes.append(MODEL_SIZES[model])
                scores.append(metrics['accuracy'])

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [MODEL_COLORS.get(m, '#1f77b4') for m in models]
    scatter = ax.scatter(sizes, scores, s=200, c=colors, edgecolors='black', linewidths=1.5, alpha=0.8)

    # Add model labels
    for model, size, score in zip(models, sizes, scores):
        label = MODEL_DISPLAY_NAMES.get(model, model)
        ax.annotate(label, (size, score), xytext=(8, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_xlabel('Model Parameters (millions)')
    ax.set_ylabel('SS3 Accuracy')
    ax.set_title('Model Size vs Secondary Structure Prediction Accuracy')
    ax.set_xscale('log')
    ax.set_ylim(0.4, 0.95)

    # Add trend line for ESM2 models
    esm2_models = ['esm2_t6_8M', 'esm2_t12_35M', 'esm2_t33_650M']
    esm2_sizes = [MODEL_SIZES[m] for m in esm2_models if m in models]
    esm2_scores = [results['ss3'][m]['accuracy'] for m in esm2_models if m in models]

    if len(esm2_sizes) > 1:
        z = np.polyfit(np.log10(esm2_sizes), esm2_scores, 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(min(esm2_sizes)), np.log10(max(esm2_sizes)), 100)
        ax.plot(x_line, p(np.log10(x_line)), '--', color='gray', alpha=0.7, label='ESM2 trend')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_ppi_comparison(results: Dict, output_path: Path):
    """Create bar chart for PPI results."""
    if 'ppi' not in results:
        return

    ppi_data = results['ppi']

    models = []
    auroc_vals = []
    auprc_vals = []

    for model, metrics in ppi_data.items():
        if isinstance(metrics, dict) and 'auroc' in metrics:
            models.append(MODEL_DISPLAY_NAMES.get(model, model))
            auroc_vals.append(metrics['auroc'])
            auprc_vals.append(metrics['auprc'])

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, auroc_vals, width, label='AUROC', color='#1976D2')
    bars2 = ax.bar(x + width/2, auprc_vals, width, label='AUPRC', color='#388E3C')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Protein-Protein Interaction Prediction')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(results: Dict, output_path: Path):
    """Create summary table as an image."""
    # Build data for table
    rows = []

    # Define task-metric pairs
    task_metrics = {
        'ss3': ('SS3 Prediction', 'accuracy', 'Accuracy'),
        'go_multilabel': ('GO Term Prediction', 'f1_max', 'F1-max'),
        'ppi': ('PPI Prediction', 'auroc', 'AUROC'),
        'variant': ('Variant Effect', 'spearman', 'Spearman'),
        'family': ('Family Classification', 'accuracy', 'Accuracy'),
        'homology': ('Remote Homology', 'accuracy', 'Accuracy'),
    }

    # Get all models
    all_models = set()
    for task_key, task_data in results.items():
        if isinstance(task_data, dict):
            all_models.update(task_data.keys())

    # Remove non-model keys
    all_models = {m for m in all_models if m in MODEL_DISPLAY_NAMES}

    # Build rows
    for task_key, (task_name, metric_key, metric_name) in task_metrics.items():
        if task_key in results:
            task_data = results[task_key]
            row = {'Task': task_name, 'Metric': metric_name}

            for model in sorted(all_models):
                display_name = MODEL_DISPLAY_NAMES.get(model, model)
                if model in task_data and isinstance(task_data[model], dict):
                    value = task_data[model].get(metric_key, np.nan)
                    row[display_name] = f'{value:.3f}' if not np.isnan(value) else '-'
                else:
                    row[display_name] = '-'

            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Create table figure
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D9E2F3')

    plt.title('ProCap Benchmark Results Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_radar_chart(results: Dict, output_path: Path):
    """Create radar chart comparing models across tasks."""
    # Collect normalized scores
    tasks = ['SS3', 'GO Terms', 'PPI', 'Variant', 'Family', 'Homology']
    task_keys = ['ss3', 'go_multilabel', 'ppi', 'variant', 'family', 'homology']
    metric_keys = ['accuracy', 'f1_max', 'auroc', 'spearman', 'accuracy', 'accuracy']

    # Get models that have all tasks
    model_scores = {}

    for task_key, metric_key, task_name in zip(task_keys, metric_keys, tasks):
        if task_key in results:
            task_data = results[task_key]
            for model, metrics in task_data.items():
                if isinstance(metrics, dict) and metric_key in metrics:
                    display_name = MODEL_DISPLAY_NAMES.get(model, model)
                    if display_name not in model_scores:
                        model_scores[display_name] = {}
                    model_scores[display_name][task_name] = metrics[metric_key]

    # Filter models with at least 3 tasks
    model_scores = {k: v for k, v in model_scores.items() if len(v) >= 3}

    if len(model_scores) < 2:
        print("Not enough models for radar chart")
        return

    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Use only tasks that all models have
    common_tasks = set.intersection(*[set(v.keys()) for v in model_scores.values()])
    tasks_to_use = [t for t in tasks if t in common_tasks]

    if len(tasks_to_use) < 3:
        print("Not enough common tasks for radar chart")
        return

    angles = np.linspace(0, 2 * np.pi, len(tasks_to_use), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    for model, scores in model_scores.items():
        values = [scores.get(task, 0) for task in tasks_to_use]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks_to_use)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Performance Across Tasks', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all benchmark visualizations."""
    # Paths
    results_dir = Path('./mparsa/bench/procap-v2/benchmark_results')
    figures_dir = Path('./mparsa/bench/procap-v2/paper/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ProCap Benchmark Visualization Generator")
    print("=" * 70)

    # Load all results
    print("\nLoading benchmark results...")
    results = load_all_results(results_dir)
    print(f"Loaded results for: {list(results.keys())}")

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. SS3 comparison chart
    print("\n1. SS3 Model Comparison Chart")
    create_ss3_comparison_chart(results, figures_dir / 'ss3_model_comparison.png')

    # 2. SS3 per-class chart
    print("\n2. SS3 Per-Class Accuracy Chart")
    create_ss3_per_class_chart(results, figures_dir / 'ss3_per_class_accuracy.png')

    # 3. Task heatmap
    print("\n3. Task Performance Heatmap")
    create_task_heatmap(results, figures_dir / 'task_performance_heatmap.png')

    # 4. Model size vs performance
    print("\n4. Model Size vs Performance")
    create_model_size_vs_performance(results, figures_dir / 'model_size_vs_performance.png')

    # 5. PPI comparison
    print("\n5. PPI Comparison Chart")
    create_ppi_comparison(results, figures_dir / 'ppi_comparison.png')

    # 6. Summary table
    print("\n6. Summary Table")
    create_summary_table(results, figures_dir / 'summary_table.png')

    # 7. Radar chart
    print("\n7. Radar Chart")
    create_radar_chart(results, figures_dir / 'model_radar_chart.png')

    print("\n" + "=" * 70)
    print(f"All visualizations saved to: {figures_dir}")
    print("=" * 70)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(figures_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
