"""
Task schemas and data models for ProCap benchmark.

Defines the structure for benchmark tasks organized by Bloom's Taxonomy levels.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class BloomLevel(str, Enum):
    """
    Bloom's Taxonomy levels for cognitive assessment.

    The ProCap benchmark evaluates PLMs at each level:
    - REMEMBERING: Recall/identify known information (GO term identification)
    - UNDERSTANDING: Explain/interpret meaning (function explanation)
    - APPLYING: Use knowledge in new situations (GO prediction)
    - ANALYSIS: Examine and break down information (variant effect prediction)
    - EVALUATION: Judge/assess quality (annotation validation)
    - CREATING: Generate new ideas/products (novel function generation)
    """

    REMEMBERING = "Remembering"
    UNDERSTANDING = "Understanding"
    APPLYING = "Applying"
    ANALYSIS = "Analysis"
    EVALUATION = "Evaluation"
    CREATING = "Creating"


class TaskType(str, Enum):
    """
    Task types supported by the benchmark.
    """

    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    SEQUENCE_CLASSIFICATION = "sequence_classification"  # Per-residue
    TOKEN_CLASSIFICATION = "token_classification"  # Per-residue (alias)
    PPI_CLASSIFICATION = "ppi_classification"  # Paired sequence
    HOMOLOGY_DETECTION = "homology_detection"  # Paired sequence homology
    REGRESSION = "regression"
    TEXT_GENERATION = "text_generation"
    SEQUENCE_GENERATION = "sequence_generation"
    STRUCTURE_PREDICTION = "structure_prediction"


class TaskConfig(BaseModel):
    """
    Configuration for a benchmark task.

    Example YAML:
        name: go_term_identification
        bloom_level: Remembering
        domain: Functional Annotation
        description: Identify GO terms for proteins from sequence
        task_type: multilabel_classification
        dataset_path: data/functional/gobench_remember.csv
        input_fields: [sequence]
        target_field: go_terms
        metrics: [f1_max, auprc_micro]
        compatible_models: [ESM-2, ProtBERT, OntoProtein]
    """

    # Task identification
    name: str = Field(..., description="Unique task identifier")
    bloom_level: BloomLevel = Field(..., description="Bloom's Taxonomy level")
    domain: str = Field(..., description="Domain (Functional, Structural, etc.)")
    description: str = Field(..., description="Task description")

    # Task type and data
    task_type: TaskType = Field(..., description="Type of task")
    dataset_path: str = Field(..., description="Path to dataset file")
    input_fields: List[str] = Field(
        default=["sequence"],
        description="Column names for input data",
    )
    target_field: str = Field(..., description="Column name for target/label")

    # Evaluation
    metrics: List[str] = Field(..., description="Metrics to compute")
    compatible_models: List[str] = Field(
        default_factory=list,
        description="Models compatible with this task",
    )

    # Task-specific settings
    num_labels: Optional[int] = Field(
        None,
        description="Number of labels (for classification)",
    )
    label_vocabulary_path: Optional[str] = Field(
        None,
        description="Path to label vocabulary file",
    )
    max_sequence_length: Optional[int] = Field(
        None,
        description="Maximum sequence length for this task",
    )
    batch_size: int = Field(
        32,
        description="Default batch size",
    )

    # Additional parameters
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task-specific parameters",
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: List[str], info) -> List[str]:
        """Validate that metrics are appropriate for task type."""
        # Get task_type from the data being validated
        task_type = info.data.get("task_type")

        # Define valid metrics per task type
        classification_metrics = {
            "accuracy",
            "f1",
            "f1_micro",
            "f1_macro",
            "f1_max",
            "precision",
            "recall",
            "precision_micro",
            "recall_micro",
            "auprc",
            "auprc_micro",
            "auprc_macro",
            "auroc",
            "auroc_micro",
            "auroc_macro",
            "mcc",
        }
        regression_metrics = {
            "spearman",
            "pearson",
            "r2",
            "rmse",
            "mae",
            "mse",
            "kendall_tau",
        }
        generation_metrics = {
            "bleu",
            "rouge",
            "rougeL",
            "bertscore",
            "novelty_50",
            "perplexity",
            "diversity",
            "seq_identity",
        }
        fairness_metrics = {
            "representation_bias",
            "group_f1",
            "group_gap",
        }

        # Combine all valid metrics
        all_valid = (
            classification_metrics
            | regression_metrics
            | generation_metrics
            | fairness_metrics
        )

        # Validate
        invalid = [m for m in v if m not in all_valid]
        if invalid:
            raise ValueError(f"Invalid metrics: {invalid}")

        return v

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


# Pre-defined task templates
DEFAULT_TASKS: Dict[str, Dict[str, Any]] = {
    # Remembering level
    "go_term_identification": {
        "name": "go_term_identification",
        "bloom_level": BloomLevel.REMEMBERING,
        "domain": "Functional Annotation",
        "description": "Identify GO terms for proteins from sequence",
        "task_type": TaskType.MULTILABEL_CLASSIFICATION,
        "dataset_path": "data/functional/gobench_remember.csv",
        "input_fields": ["sequence"],
        "target_field": "go_terms",
        "metrics": ["f1_max", "auprc_micro", "auroc_micro"],
        "compatible_models": ["esm2_t33_650M", "protbert", "ontoprotein"],
    },
    # Understanding level
    "function_explanation": {
        "name": "function_explanation",
        "bloom_level": BloomLevel.UNDERSTANDING,
        "domain": "Functional Annotation",
        "description": "Generate free-text description of protein function",
        "task_type": TaskType.TEXT_GENERATION,
        "dataset_path": "data/functional/gobench_understand.csv",
        "input_fields": ["sequence"],
        "target_field": "function_text",
        "metrics": ["rougeL", "bertscore", "novelty_50"],
        "compatible_models": ["protllm", "progen2"],
    },
    # Applying level
    "go_term_prediction": {
        "name": "go_term_prediction",
        "bloom_level": BloomLevel.APPLYING,
        "domain": "Functional Annotation",
        "description": "Predict GO terms for unannotated proteins",
        "task_type": TaskType.MULTILABEL_CLASSIFICATION,
        "dataset_path": "data/functional/gobench_apply.csv",
        "input_fields": ["sequence"],
        "target_field": "go_terms",
        "metrics": ["f1_max", "auprc_micro", "representation_bias"],
        "compatible_models": ["esm2_t33_650M", "protbert", "ontoprotein"],
    },
    # Analysis level
    "variant_effect_prediction": {
        "name": "variant_effect_prediction",
        "bloom_level": BloomLevel.ANALYSIS,
        "domain": "Functional Annotation",
        "description": "Predict effect of sequence variants on function",
        "task_type": TaskType.REGRESSION,
        "dataset_path": "data/functional/proteingym_variant.csv",
        "input_fields": ["sequence", "variant"],
        "target_field": "effect_score",
        "metrics": ["spearman", "rmse"],
        "compatible_models": ["esm2_t33_650M", "protbert"],
    },
    # Evaluation level
    "annotation_validation": {
        "name": "annotation_validation",
        "bloom_level": BloomLevel.EVALUATION,
        "domain": "Functional Annotation",
        "description": "Validate predicted GO annotations",
        "task_type": TaskType.MULTILABEL_CLASSIFICATION,
        "dataset_path": "data/functional/peer_eval.csv",
        "input_fields": ["sequence"],
        "target_field": "go_terms",
        "metrics": ["precision_micro", "recall_micro", "f1_max"],
        "compatible_models": ["esm2_t33_650M", "protbert", "ontoprotein"],
    },
    # Creating level
    "novel_function_generation": {
        "name": "novel_function_generation",
        "bloom_level": BloomLevel.CREATING,
        "domain": "Functional Annotation",
        "description": "Generate novel functional categories",
        "task_type": TaskType.TEXT_GENERATION,
        "dataset_path": "data/functional/gobench_create.csv",
        "input_fields": ["sequence"],
        "target_field": "new_category_text",
        "metrics": ["diversity", "novelty_50"],
        "compatible_models": ["protllm", "progen2"],
    },
}
