# ProCap: Protein Capability Benchmark

ProCap is a comprehensive benchmark for evaluating Protein Language Models (PLMs) across cognitive ability levels based on Bloom's Taxonomy.

## Installation

```bash
# Clone the repository
git clone https://github.com/msparsa/procap.git
cd procap-v2

# Install dependencies (recommended)
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Data Setup

Most benchmark datasets are included in the repository. However, the ProteinGym variant effect dataset (~2GB) must be downloaded separately:

```bash
# Download large data files (ProteinGym ~2GB)
python scripts/download_data.py

# List available downloads without downloading
python scripts/download_data.py --list
```

**Included datasets:**
- PPI prediction (53MB) - tracked with git-lfs
- GO term identification (20MB)
- Secondary structure SS3 (11MB)
- Remote homology (8.9MB)
- Family classification (3.5MB)
- All `*_small.csv` test datasets (<2MB each)

**External datasets (download required):**
- ProteinGym substitutions (2GB) - for variant effect prediction

See `configs/data_sources.yaml` for dataset sources and citations.

## Running Benchmarks

### Quick Single-Model Tests

```bash
# Run SS3 benchmark on a single model
python run_ss3_benchmark.py

# Run GO term benchmark
python run_go_benchmark.py

# Run PPI benchmark
python run_ppi_benchmark.py

# Run other benchmarks
python run_family_benchmark.py
python run_homology_benchmark.py
python run_variant_benchmark.py
```

### Full Benchmark Suite (All Models)

```bash
# Run all models on each task
python run_ss3_all_models.py
python run_go_all_models.py
python run_ppi_all_models.py
python run_family_all_models.py
python run_homology_all_models.py
python run_variant_all_models.py
```

Results are saved to `benchmark_results/` as JSON files.

## Quick Start

```bash
# Run a quick model test
python -m procap.cli test --model esm2_t6_8M

# List all available models
python -m procap.cli list-models

# List all benchmark tasks
python -m procap.cli list-tasks

# Get task info
python -m procap.cli info --task secondary_structure_ss3

# Run evaluation
python -m procap.cli evaluate --model esm2_t33_650M --task secondary_structure_ss3 --max-samples 100
```

## Supported Models

ProCap supports **37 model variants** across **11 model families**:

### Encoder Models

| Model Family | Variants | Parameters | Description |
|--------------|----------|------------|-------------|
| **ESM-2** | 6 variants | 8M - 15B | Meta AI's state-of-the-art PLM |
| **ESM-1b/1v** | 6 variants | 650M | Earlier ESM versions, ESM-1v optimized for variants |
| **ProtBERT** | 2 variants | ~420M | BERT-based PLM from Rostlab |
| **OntoProtein** | 1 variant | ~420M | GO knowledge-enhanced PLM |
| **ProtT5** | 3 variants | ~3B | T5-based PLM from Rostlab |
| **Ankh** | 2 variants | 450M - 1.5B | Efficient PLM from ElnaggarLab |
| **ProtAlbert** | 1 variant | 12M | Lightweight ALBERT-based PLM |
| **SaProt** | 2 variants | 650M | Structure-aware PLM (ICLR 2024) |

### Generative Models

| Model Family | Variants | Parameters | Description |
|--------------|----------|------------|-------------|
| **ProGen2** | 4 variants | 151M - 2.7B | Protein sequence generation |
| **ZymCTRL** | 1 variant | ~350M | Enzyme-specialized generation |
| **ProtGPT2** | 1 variant | 738M | GPT-2 based protein generation |
| **Tranception** | 3 variants | Various | Mutation effect prediction specialist |

### Tested Models (Linear Probe Benchmarks)

The following 11 models have been extensively tested across all benchmark tasks:

| Model | Parameters | Status |
|-------|------------|--------|
| ESM2-8M | 8M | Working |
| ESM2-35M | 35M | Working |
| ESM2-150M | 150M | Working |
| ESM2-650M | 650M | Working |
| ESM1b-650M | 650M | Working |
| ProtBERT | 420M | Working |
| ProtBERT-BFD | 420M | Working |
| OntoProtein | 420M | Working |
| ProtT5-XL-BFD | 3B | Working |
| Ankh-Base | 450M | Working |
| ProtAlbert | 12M | Working |

**Note:** ProGen2 and ProtGPT2 have known compatibility issues with linear probe evaluation.

### Model Usage Examples

```python
from procap.models import get_model

# Load ESM-2
model = get_model("esm2_t33_650M")
model.load()
embeddings = model.get_embeddings(["MLKFV", "ACDEFG"], pooling="mean")

# Load SaProt (structure-aware)
model = get_model("saprot_650m_af2")
model.load()
# With structure tokens
embeddings = model.get_embeddings(["MaLbKcFdVe"])
# Or sequence-only (auto-adds default structure tokens)
embeddings = model.get_embeddings(["MLKFV"])

# Load ProtGPT2 for generation
model = get_model("protgpt2")
model.load()
generated = model.generate(["MKTV"], max_new_tokens=50)

# Load Tranception for mutation scoring
model = get_model("tranception_medium")
model.load()
scores = model.score_mutations("MLKFVAAAA", ["M1A", "L2V"])
```

## Benchmark Tasks

ProCap includes **11 tasks** organized by Bloom's Taxonomy levels:

### Task Summary

| Task | Type | Bloom Level | Dataset Size |
|------|------|-------------|--------------|
| `go_term_identification` | multilabel_classification | Remembering | 30,184 |
| `family_classification` | multiclass_classification | Remembering | 6,372 |
| `function_explanation` | text_generation | Understanding | 4* |
| `secondary_structure_ss3` | token_classification | Understanding | 10,000 |
| `go_term_prediction` | multilabel_classification | Applying | 5,000 |
| `variant_effect_prediction` | regression | Analysis | 2,465,767 |
| `remote_homology` | multiclass_classification | Analysis | 19,983 |
| `ppi_prediction` | ppi_classification | Analysis | 50,000 |
| `annotation_validation` | multilabel_classification | Evaluation | 2,000 |
| `novel_function_generation` | text_generation | Creating | 1,000 |
| `enzyme_sequence_generation` | sequence_generation | Creating | 1,000 |

*function_explanation has placeholder data

### Task Types

- **multilabel_classification**: Predict multiple labels per sample (e.g., GO terms)
- **multiclass_classification**: Single class prediction (e.g., protein family)
- **token_classification**: Per-residue prediction (e.g., secondary structure)
- **regression**: Continuous value prediction (e.g., variant effects)
- **ppi_classification**: Paired sequence classification (protein interactions)
- **text_generation**: Generate text descriptions
- **sequence_generation**: Generate protein sequences

## CLI Reference

### Commands

```bash
# Test model loading and basic inference
python -m procap.cli test [--model MODEL_NAME]

# List available models
python -m procap.cli list-models

# List available tasks (optionally filter by Bloom level)
python -m procap.cli list-tasks [--bloom LEVEL]

# Get detailed info about a model or task
python -m procap.cli info --model MODEL_NAME
python -m procap.cli info --task TASK_NAME

# Run evaluation
python -m procap.cli evaluate \
    --model MODEL_NAME \
    --task TASK_NAME \
    [--max-samples N] \
    [--batch-size B] \
    [--output results.json] \
    [--device cuda|cpu]
```

### Examples

```bash
# Evaluate ESM-2 on secondary structure prediction
python -m procap.cli evaluate \
    --model esm2_t33_650M \
    --task secondary_structure_ss3 \
    --max-samples 1000 \
    --batch-size 8

# Evaluate on GO term prediction
python -m procap.cli evaluate \
    --model protbert \
    --task go_term_prediction \
    --max-samples 500

# Evaluate on protein-protein interaction
python -m procap.cli evaluate \
    --model esm2_t33_650M \
    --task ppi_prediction \
    --max-samples 1000
```

## Python API

### Basic Usage

```python
from procap.models import get_model
from procap.tasks import get_task
from procap.data import load_dataset
from procap.runners import MultiLabelClassificationRunner

# Load model
model = get_model("esm2_t33_650M")
model.load()

# Load task configuration
task = get_task("go_term_identification")

# Load dataset
df = load_dataset(task.dataset_path, nrows=1000)

# Run evaluation
runner = MultiLabelClassificationRunner(task, model, batch_size=16)
results = runner.run(df)
print(results)
```

### Custom Evaluation

```python
from procap.models import get_model

# Get embeddings for downstream tasks
model = get_model("esm2_t33_650M")
model.load()

sequences = ["MLKFVAAAA", "ACDEFGHIK", "MKTVRQERLK"]
embeddings = model.get_embeddings(sequences, pooling="mean")

# embeddings shape: [3, hidden_size]
print(f"Embedding shape: {embeddings.shape}")
```

## Data Directory Structure

```
data/
├── functional/
│   ├── gobench_remember.csv      # GO term identification
│   ├── gobench_understand.csv    # Function explanation
│   ├── gobench_apply.csv         # GO term prediction
│   ├── gobench_create.csv        # Novel function generation
│   ├── family_classification.csv # Protein family
│   └── peer_eval.csv             # Annotation validation
├── variant_effect/
│   └── proteingym_substitutions.csv  # Variant effect prediction
├── structure/
│   ├── secondary_structure_ss3.csv      # SS3 prediction (10K)
│   └── secondary_structure_ss3_small.csv # SS3 small (1K)
├── homology/
│   └── remote_homology.csv       # Remote homology detection
└── interaction/
    └── ppi_prediction.csv        # Protein-protein interaction
```

## Metrics

### Classification Metrics
- `accuracy`, `f1_micro`, `f1_macro`, `f1_max`
- `precision_micro`, `recall_micro`
- `auprc_micro`, `auprc_macro`
- `auroc_micro`, `auroc_macro`

### Regression Metrics
- `spearman`, `pearson`, `kendall_tau`
- `rmse`, `mae`, `r2`

### Generation Metrics
- `rougeL`, `bertscore`
- `novelty_50`, `diversity`
- `seq_identity`

## Configuration

### Task Configuration (configs/tasks.yaml)

```yaml
secondary_structure_ss3:
  name: secondary_structure_ss3
  bloom_level: Understanding
  domain: Structure
  description: Predict 3-state secondary structure for each residue
  task_type: token_classification
  dataset_path: data/structure/secondary_structure_ss3.csv
  input_fields: [sequence]
  target_field: ss3_labels
  metrics: [accuracy, f1_macro]
  num_labels: 3
```

### Model Configuration (configs/models.yaml)

```yaml
models:
  my_custom_esm:
    type: esm
    variant: esm2_t33_650M
    max_batch_size: 8
```

## Adding New Models

1. Create adapter in `procap/models/`:

```python
from procap.core.base_model import BaseProteinModel, ModelOutput

class MyModelAdapter(BaseProteinModel):
    def load(self) -> None:
        # Load model weights
        pass
    
    def tokenize(self, sequences):
        # Tokenize sequences
        pass
    
    def get_embeddings(self, sequences, pooling="mean"):
        # Extract embeddings
        pass
```

2. Register in `procap/models/registry.py`:

```python
from procap.models.my_model import MyModelAdapter

DEFAULT_MODELS["my_model"] = {
    "class": MyModelAdapter,
    "params": {"variant": "my_variant", "max_batch_size": 8},
}
```

<!-- ## Citation

If you use ProCap in your research, please cite:

```bibtex
@misc{procap2024,
  title={ProCap: A Protein Capability Benchmark},
  year={2024},
}
```

## License

MIT License -->
