#!/usr/bin/env python
"""
Create visualizations for ProCap benchmark paper.

Generates publication-ready figures for benchmark results.
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

# Color palette for models
MODEL_COLORS = {
    'esm2_t6_8M': '#1f77b4',
    'esm2_t12_35M': '#2ca02c',
    'esm2_t30_150M': '#ff7f0e',
    'esm2_t33_650M': '#d62728',
    'esm2_t36_3B': '#9467bd',
    'protbert': '#8c564b',
    'protbert_bfd': '#e377c2',
    'prott5_xl_bfd': '#7f7f7f',
    'ankh_base': '#bcbd22',
    'ankh_large': '#17becf',
    'saprot_650m_af2': '#ff9896',
    'progen2_small': '#aec7e8',
}

# Bloom's Taxonomy colors
BLOOM_COLORS = {
    'Remembering': '#E3F2FD',
    'Understanding': '#BBDEFB',
    'Applying': '#90CAF9',
    'Analysis': '#64B5F6',
    'Evaluation': '#42A5F5',
    'Creating': '#2196F3',
}


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load all benchmark results from directory."""
    results = {}
    for json_file in results_dir.glob('*.json'):
        with open(json_file) as f:
            results[json_file.stem] = json.load(f)
    return results


def create_model_comparison_bar_chart(
    results: Dict[str, Dict],
    metric: str = 'accuracy',
    title: str = 'Model Performance Comparison',
    output_path: Path = None,
):
    """Create bar chart comparing models on a single metric."""
    models = []
    scores = []

    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and metric in model_results.get('metrics', {}):
            models.append(model_name)
            scores.append(model_results['metrics'][metric])

    if not models:
        print(f"No results found for metric: {metric}")
        return

    # Sort by score
    sorted_data = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
    models, scores = zip(*sorted_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [MODEL_COLORS.get(m, '#1f77b4') for m in models]
    bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Model')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_ylim(0, max(scores) * 1.15)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def create_bloom_taxonomy_radar(
    results_by_bloom: Dict[str, float],
    model_name: str,
    output_path: Path = None,
):
    """Create radar chart showing model performance across Bloom levels."""
    levels = ['Remembering', 'Understanding', 'Applying', 'Analysis', 'Evaluation', 'Creating']

    # Filter to only levels with results
    available_levels = [l for l in levels if l in results_by_bloom]
    values = [results_by_bloom[l] for l in available_levels]

    if len(available_levels) < 3:
        print(f"Not enough Bloom levels for radar chart (need >= 3, have {len(available_levels)})")
        return

    # Close the radar chart
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(available_levels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color='#2196F3')
    ax.fill(angles, values, alpha=0.25, color='#2196F3')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_levels)
    ax.set_ylim(0, 1)
    ax.set_title(f'{model_name} Performance Across Bloom\'s Taxonomy', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def create_task_heatmap(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    output_path: Path = None,
):
    """Create heatmap of model x task performance."""
    # Convert to DataFrame
    models = sorted(results.keys())
    tasks = set()
    for model_results in results.values():
        tasks.update(model_results.keys())
    tasks = sorted(tasks)

    data = []
    for model in models:
        row = []
        for task in tasks:
            if task in results[model]:
                score = results[model][task].get(metric, np.nan)
            else:
                score = np.nan
            row.append(score)
        data.append(row)

    df = pd.DataFrame(data, index=models, columns=tasks)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )

    ax.set_title(f'Model Performance Across Tasks ({metric})')
    ax.set_xlabel('Task')
    ax.set_ylabel('Model')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def create_ss3_per_class_chart(
    results: Dict[str, Dict],
    output_path: Path = None,
):
    """Create grouped bar chart for SS3 per-class accuracy."""
    models = []
    h_acc = []
    e_acc = []
    c_acc = []

    for model_name, model_results in results.items():
        metrics = model_results.get('metrics', {})
        if 'accuracy_H' in metrics:
            models.append(model_name.replace('esm2_', 'ESM2-').replace('protbert', 'ProtBERT'))
            h_acc.append(metrics['accuracy_H'])
            e_acc.append(metrics['accuracy_E'])
            c_acc.append(metrics['accuracy_C'])

    if not models:
        print("No SS3 per-class results found")
        return

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_h = ax.bar(x - width, h_acc, width, label='Helix (H)', color='#E53935')
    bars_e = ax.bar(x, e_acc, width, label='Strand (E)', color='#1E88E5')
    bars_c = ax.bar(x + width, c_acc, width, label='Coil (C)', color='#43A047')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Secondary Structure Prediction: Per-Class Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bars in [bars_h, bars_e, bars_c]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


def create_model_size_vs_performance(
    results: Dict[str, float],
    model_sizes: Dict[str, int],
    metric_name: str = 'Accuracy',
    output_path: Path = None,
):
    """Create scatter plot of model size vs performance."""
    models = []
    sizes = []
    scores = []

    for model, score in results.items():
        if model in model_sizes:
            models.append(model)
            sizes.append(model_sizes[model])
            scores.append(score)

    if not models:
        print("No matching model sizes found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        sizes, scores,
        s=100,
        c=[MODEL_COLORS.get(m, '#1f77b4') for m in models],
        edgecolors='black',
        linewidths=1,
        alpha=0.8
    )

    # Add model labels
    for model, size, score in zip(models, sizes, scores):
        label = model.replace('esm2_', 'ESM2-').replace('_', '-')
        ax.annotate(label, (size, score), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)

    ax.set_xlabel('Model Parameters (millions)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Model Size vs {metric_name}')
    ax.set_xscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()


# Model sizes in millions of parameters
MODEL_SIZES = {
    'esm2_t6_8M': 8,
    'esm2_t12_35M': 35,
    'esm2_t30_150M': 150,
    'esm2_t33_650M': 650,
    'esm2_t36_3B': 3000,
    'esm2_t48_15B': 15000,
    'protbert': 420,
    'protbert_bfd': 420,
    'prott5_xl_bfd': 3000,
    'ankh_base': 450,
    'ankh_large': 1500,
    'saprot_650m_af2': 650,
    'progen2_small': 150,
    'progen2_medium': 760,
    'protgpt2': 738,
}


def main():
    """Generate all visualizations."""
    results_dir = Path('benchmark_results')
    figures_dir = Path('paper/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(results_dir)

    if not results:
        print("No benchmark results found in benchmark_results/")
        print("Creating example visualizations with placeholder data...")

        # Create placeholder data for demonstration
        placeholder_ss3 = {
            'esm2_t6_8M': {'metrics': {'accuracy': 0.759, 'f1_macro': 0.722, 'accuracy_H': 0.825, 'accuracy_E': 0.512, 'accuracy_C': 0.784}},
            'esm2_t33_650M': {'metrics': {'accuracy': 0.823, 'f1_macro': 0.791, 'accuracy_H': 0.872, 'accuracy_E': 0.641, 'accuracy_C': 0.812}},
            'protbert': {'metrics': {'accuracy': 0.729, 'f1_macro': 0.694, 'accuracy_H': 0.859, 'accuracy_E': 0.510, 'accuracy_C': 0.669}},
            'protbert_bfd': {'metrics': {'accuracy': 0.751, 'f1_macro': 0.712, 'accuracy_H': 0.861, 'accuracy_E': 0.558, 'accuracy_C': 0.698}},
        }

        # Create SS3 comparison chart
        create_model_comparison_bar_chart(
            placeholder_ss3,
            metric='accuracy',
            title='Secondary Structure Prediction (SS3) - Overall Accuracy',
            output_path=figures_dir / 'ss3_accuracy_comparison.png'
        )

        # Create SS3 per-class chart
        create_ss3_per_class_chart(
            placeholder_ss3,
            output_path=figures_dir / 'ss3_per_class_accuracy.png'
        )

        # Model size vs performance
        size_results = {m: r['metrics']['accuracy'] for m, r in placeholder_ss3.items()}
        create_model_size_vs_performance(
            size_results,
            MODEL_SIZES,
            metric_name='SS3 Accuracy',
            output_path=figures_dir / 'model_size_vs_accuracy.png'
        )

        print(f"\nExample visualizations saved to {figures_dir}/")
        return

    # Process actual results
    print(f"Found {len(results)} result files")

    # SS3 results
    if 'ss3_results' in results:
        ss3_data = results['ss3_results']

        create_model_comparison_bar_chart(
            ss3_data,
            metric='accuracy',
            title='Secondary Structure Prediction (SS3) - Overall Accuracy',
            output_path=figures_dir / 'ss3_accuracy_comparison.png'
        )

        create_ss3_per_class_chart(
            ss3_data,
            output_path=figures_dir / 'ss3_per_class_accuracy.png'
        )

    print(f"\nAll visualizations saved to {figures_dir}/")


if __name__ == '__main__':
    main()
