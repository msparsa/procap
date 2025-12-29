#!/usr/bin/env python3
"""
Create publication-ready visualizations for the ProCap benchmark paper.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.grid': False
})

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Define paths
BASE_DIR = Path("./mparsa/bench/procap-v2")
RESULTS_DIR = BASE_DIR / "benchmark_results"
FIGURES_DIR = BASE_DIR / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color palettes for model families
MODEL_COLORS = {
    'ESM-2': '#3498db',  # Blue
    'ProtBERT': '#e74c3c',  # Red
    'ProtTrans': '#2ecc71',  # Green
    'ESM-1b': '#9b59b6',  # Purple
    'TAPE': '#f39c12',  # Orange
}

# Model parameters (in millions)
MODEL_PARAMS = {
    'esm2_t6_8M': 8,
    'esm2_t12_35M': 35,
    'esm2_t30_150M': 150,
    'esm2_t33_650M': 650,
    'esm2_t36_3B': 3000,
    'esm2_t48_15B': 15000,
    'protbert': 420,
    'protbert_bfd': 420,
    'prottrans_t5_xl_u50': 3000,
    'esm1b_t33_650M': 650,
}

# Model display names
MODEL_DISPLAY_NAMES = {
    'esm2_t6_8M': 'ESM-2 (8M)',
    'esm2_t12_35M': 'ESM-2 (35M)',
    'esm2_t30_150M': 'ESM-2 (150M)',
    'esm2_t33_650M': 'ESM-2 (650M)',
    'esm2_t36_3B': 'ESM-2 (3B)',
    'esm2_t48_15B': 'ESM-2 (15B)',
    'protbert': 'ProtBERT',
    'protbert_bfd': 'ProtBERT-BFD',
    'prottrans_t5_xl_u50': 'ProtTrans-XL',
    'esm1b_t33_650M': 'ESM-1b',
}

def get_model_family(model_name):
    """Get model family from model name."""
    if 'esm2' in model_name.lower():
        return 'ESM-2'
    elif 'esm1' in model_name.lower():
        return 'ESM-1b'
    elif 'protbert' in model_name.lower():
        return 'ProtBERT'
    elif 'prottrans' in model_name.lower():
        return 'ProtTrans'
    else:
        return 'Other'

def load_benchmark_results():
    """Load all benchmark results from JSON files."""
    results = []

    if RESULTS_DIR.exists():
        # First check for combined ss3_results.json
        ss3_combined = RESULTS_DIR / "ss3_results.json"
        if ss3_combined.exists():
            try:
                with open(ss3_combined, 'r') as f:
                    data = json.load(f)
                    # Convert combined format to individual format
                    for model_name, metrics in data.items():
                        results.append({
                            'task': 'secondary_structure_ss3',
                            'model': model_name,
                            'metrics': metrics
                        })
            except Exception as e:
                print(f"Error loading {ss3_combined}: {e}")

        # Load individual result files
        for json_file in RESULTS_DIR.glob("ss3_*.json"):
            if json_file.name == "ss3_results.json":
                continue  # Already processed
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'task' in data:
                        results.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    return results

def create_ss3_accuracy_comparison(results, output_path):
    """Create bar chart comparing models on SS3 accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract SS3 results
    ss3_results = [r for r in results if r['task'] == 'secondary_structure_ss3']

    if ss3_results:
        models = [MODEL_DISPLAY_NAMES.get(r['model'], r['model']) for r in ss3_results]
        accuracies = [r['metrics']['accuracy'] * 100 for r in ss3_results]
        families = [get_model_family(r['model']) for r in ss3_results]
        colors = [MODEL_COLORS[f] for f in families]

        bars = ax.bar(range(len(models)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Secondary Structure (SS3) Prediction Accuracy', fontweight='bold', pad=20)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 100)

        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Create legend for model families
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=MODEL_COLORS[f], label=f, alpha=0.8, edgecolor='black')
                          for f in sorted(set(families))]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")

def create_ss3_per_class_accuracy(results, output_path):
    """Create grouped bar chart for H/E/C per-class accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract SS3 results
    ss3_results = [r for r in results if r['task'] == 'secondary_structure_ss3']

    if ss3_results:
        models = [MODEL_DISPLAY_NAMES.get(r['model'], r['model']) for r in ss3_results]

        # Extract per-class accuracies
        acc_h = [r['metrics'].get('accuracy_H', 0) * 100 for r in ss3_results]
        acc_e = [r['metrics'].get('accuracy_E', 0) * 100 for r in ss3_results]
        acc_c = [r['metrics'].get('accuracy_C', 0) * 100 for r in ss3_results]

        x = np.arange(len(models))
        width = 0.25

        bars1 = ax.bar(x - width, acc_h, width, label='Helix (H)', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x, acc_e, width, label='Strand (E)', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        bars3 = ax.bar(x + width, acc_c, width, label='Coil (C)', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Per-Class Accuracy (%)', fontweight='bold')
        ax.set_title('Secondary Structure Per-Class Accuracy (H/E/C)', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.legend(loc='upper right', framealpha=0.9)

        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                           f'{height:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")

def create_model_size_vs_accuracy(results, output_path):
    """Create scatter plot of model parameters vs performance."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Extract SS3 results with known parameter counts
    ss3_results = [r for r in results if r['task'] == 'secondary_structure_ss3']

    if ss3_results:
        data_points = []
        for r in ss3_results:
            model = r['model']
            if model in MODEL_PARAMS:
                params = MODEL_PARAMS[model]
                accuracy = r['metrics']['accuracy'] * 100
                family = get_model_family(model)
                display_name = MODEL_DISPLAY_NAMES.get(model, model)
                data_points.append((params, accuracy, family, display_name))

        if data_points:
            params_list, acc_list, families, names = zip(*data_points)

            # Plot points by family
            for family in set(families):
                mask = [f == family for f in families]
                family_params = [p for p, m in zip(params_list, mask) if m]
                family_acc = [a for a, m in zip(acc_list, mask) if m]
                family_names = [n for n, m in zip(names, mask) if m]

                ax.scatter(family_params, family_acc, s=200, alpha=0.7,
                          color=MODEL_COLORS[family], label=family,
                          edgecolors='black', linewidth=1.5)

                # Add labels for each point
                for x, y, name in zip(family_params, family_acc, family_names):
                    ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=9, alpha=0.8)

            ax.set_xscale('log')
            ax.set_xlabel('Model Parameters (millions)', fontweight='bold')
            ax.set_ylabel('SS3 Accuracy (%)', fontweight='bold')
            ax.set_title('Model Size vs. Secondary Structure Prediction Accuracy',
                        fontweight='bold', pad=20)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")

def create_bloom_taxonomy_overview(output_path):
    """Create visual showing the 6 Bloom levels with task counts."""
    # Define Bloom's taxonomy levels with ProCap tasks
    bloom_data = {
        'Remember': {
            'level': 1,
            'tasks': ['GO Function Recall', 'Family Classification'],
            'count': 2,
            'color': '#e8f4f8'
        },
        'Understand': {
            'level': 2,
            'tasks': ['GO Function Understanding', 'Remote Homology'],
            'count': 2,
            'color': '#c3e3f0'
        },
        'Apply': {
            'level': 3,
            'tasks': ['Secondary Structure (SS3/SS8)', 'Subcellular Localization'],
            'count': 2,
            'color': '#9dd4e8'
        },
        'Analyze': {
            'level': 4,
            'tasks': ['Protein-Protein Interaction', 'Variant Effect Prediction'],
            'count': 2,
            'color': '#6ec4e0'
        },
        'Evaluate': {
            'level': 5,
            'tasks': ['Peer Review Prediction'],
            'count': 1,
            'color': '#3ba9d4'
        },
        'Create': {
            'level': 6,
            'tasks': ['Protein Design', 'Function-to-Sequence'],
            'count': 2,
            'color': '#1e88b8'
        }
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    levels = sorted(bloom_data.keys(), key=lambda x: bloom_data[x]['level'])
    y_positions = list(range(len(levels)))

    # Create horizontal bars
    for i, level in enumerate(levels):
        data = bloom_data[level]
        count = data['count']
        color = data['color']

        # Draw bar
        ax.barh(i, count, height=0.6, color=color, edgecolor='black', linewidth=2, alpha=0.9)

        # Add level name (bold and larger)
        ax.text(-0.3, i, f"{data['level']}. {level}",
               fontsize=13, fontweight='bold', va='center', ha='right')

        # Add task count
        ax.text(count + 0.1, i, f"{count} tasks",
               fontsize=11, va='center', ha='left', fontweight='bold')

        # Add task examples (smaller text, inside or near the bar)
        task_text = '\n'.join(data['tasks'])
        ax.text(count/2, i, task_text,
               fontsize=9, va='center', ha='center', style='italic')

    ax.set_yticks([])
    ax.set_xlabel('Number of Tasks', fontweight='bold', fontsize=13)
    ax.set_title("ProCap Benchmark: Bloom's Taxonomy Coverage",
                fontweight='bold', pad=20, fontsize=15)
    ax.set_xlim(-0.5, max(d['count'] for d in bloom_data.values()) + 1)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Add cognitive complexity arrow
    ax.annotate('', xy=(-0.3, len(levels)-0.5), xytext=(-0.3, -0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))
    ax.text(-0.5, len(levels)/2, 'Cognitive\nComplexity',
           fontsize=11, fontweight='bold', rotation=90,
           va='center', ha='center', color='darkred')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")

def create_task_type_distribution(output_path):
    """Create pie chart of task types in the benchmark."""
    # Define task categories based on ProCap benchmark
    task_categories = {
        'Structural': 3,  # SS3, SS8, Disorder
        'Functional': 4,  # GO tasks, Family classification
        'Interaction': 2,  # PPI, Binding sites
        'Variant Effect': 1,  # ProteinGym
        'Homology': 1,  # Remote homology
        'Generative': 2,  # Protein design, F2S
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    labels = list(task_categories.keys())
    sizes = list(task_categories.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                       colors=colors, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'},
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 2})

    # Make percentage text bold and white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    # Add task counts in legend
    legend_labels = [f'{label}: {count} tasks' for label, count in task_categories.items()]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
             framealpha=0.9, fontsize=11)

    ax.set_title('ProCap Benchmark: Task Type Distribution',
                fontweight='bold', pad=20, fontsize=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path}")

def main():
    """Create all publication figures."""
    print("=" * 60)
    print("Creating Publication-Ready Visualizations for ProCap")
    print("=" * 60)
    print()

    # Load benchmark results
    print("Loading benchmark results...")
    results = load_benchmark_results()
    print(f"Loaded {len(results)} benchmark result(s)")
    print()

    # Create figures
    print("Creating figures...")
    print()

    # 1. SS3 Accuracy Comparison
    output_path = FIGURES_DIR / "ss3_accuracy_comparison.png"
    create_ss3_accuracy_comparison(results, output_path)

    # 2. SS3 Per-Class Accuracy
    output_path = FIGURES_DIR / "ss3_per_class_accuracy.png"
    create_ss3_per_class_accuracy(results, output_path)

    # 3. Model Size vs Accuracy
    output_path = FIGURES_DIR / "model_size_vs_accuracy.png"
    create_model_size_vs_accuracy(results, output_path)

    # 4. Bloom Taxonomy Overview
    output_path = FIGURES_DIR / "bloom_taxonomy_overview.png"
    create_bloom_taxonomy_overview(output_path)

    # 5. Task Type Distribution
    output_path = FIGURES_DIR / "task_type_distribution.png"
    create_task_type_distribution(output_path)

    print()
    print("=" * 60)
    print("All figures created successfully!")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
