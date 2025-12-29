"""
ProCap Benchmark CLI.

Usage:
    procap evaluate --model esm2_t33_650M --task go_term_identification
    procap list-models
    procap list-tasks
    procap info --model esm2_t33_650M
"""

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="procap",
    help="ProCap: Protein Capability Benchmark for evaluating Protein Language Models",
    add_completion=False,
)

console = Console()


@app.command()
def evaluate(
    model: str = typer.Option(..., "--model", "-m", help="Model name to evaluate"),
    task: str = typer.Option(..., "--task", "-t", help="Task name to run"),
    data_path: Optional[Path] = typer.Option(
        None, "--data", "-d", help="Path to data file (overrides task default)"
    ),
    output: Path = typer.Option(
        Path("results.json"), "--output", "-o", help="Output file for results"
    ),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size"),
    max_samples: Optional[int] = typer.Option(
        None, "--max-samples", "-n", help="Maximum samples to evaluate"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device (cuda/cpu)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Run benchmark evaluation for a model on a task.

    Example:
        procap evaluate --model esm2_t33_650M --task go_term_identification
    """
    from procap.models.registry import get_model
    from procap.tasks.registry import get_task
    from procap.data.loader import load_dataset
    from procap.runners.classification import (
        MultiLabelClassificationRunner,
        MulticlassClassificationRunner,
        BinaryClassificationRunner,
    )
    from procap.runners.regression import RegressionRunner
    from procap.runners.token_classification import TokenClassificationRunner
    from procap.runners.generation import TextGenerationRunner, SequenceGenerationRunner
    from procap.runners.ppi import PPIClassificationRunner
    from procap.tasks.schemas import TaskType

    console.print(f"[bold blue]ProCap Benchmark[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Task: {task}")
    console.print()

    # Load model
    console.print("[yellow]Loading model...[/yellow]")
    try:
        plm = get_model(model, device=device)
        plm.load()
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Model loaded: {plm.name}[/green]")
    console.print(f"  Hidden size: {plm.hidden_size}")
    console.print(f"  Device: {plm.device}")
    console.print()

    # Load task
    console.print("[yellow]Loading task...[/yellow]")
    try:
        task_config = get_task(task)
    except Exception as e:
        console.print(f"[red]Error loading task: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Task loaded: {task_config.name}[/green]")
    console.print(f"  Bloom level: {task_config.bloom_level}")
    console.print(f"  Type: {task_config.task_type}")
    console.print()

    # Load data
    data_file = data_path or Path(task_config.dataset_path)
    console.print(f"[yellow]Loading data from {data_file}...[/yellow]")

    if not data_file.exists():
        console.print(f"[red]Data file not found: {data_file}[/red]")
        console.print("[yellow]Tip: Create sample data or specify --data path[/yellow]")
        raise typer.Exit(1)

    try:
        df = load_dataset(
            data_file,
            input_columns=task_config.input_fields,
            target_column=task_config.target_field,
            nrows=max_samples,
        )
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Loaded {len(df)} samples[/green]")
    console.print()

    # Select runner based on task type
    task_type = TaskType(task_config.task_type)

    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        runner = MultiLabelClassificationRunner(
            task_config, plm, batch_size=batch_size, show_progress=True
        )
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        runner = MulticlassClassificationRunner(
            task_config, plm, batch_size=batch_size, show_progress=True
        )
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        runner = BinaryClassificationRunner(
            task_config, plm, batch_size=batch_size, show_progress=True
        )
    elif task_type == TaskType.REGRESSION:
        runner = RegressionRunner(
            task_config, plm, batch_size=batch_size, show_progress=True
        )
    elif task_type in [TaskType.TOKEN_CLASSIFICATION, TaskType.SEQUENCE_CLASSIFICATION]:
        # Get num_classes from task config (e.g., 3 for SS3, 8 for SS8)
        num_classes = task_config.num_labels or 3
        runner = TokenClassificationRunner(
            task_config, plm, batch_size=batch_size, show_progress=True, num_classes=num_classes
        )
    elif task_type == TaskType.TEXT_GENERATION:
        runner = TextGenerationRunner(
            task_config, plm, batch_size=batch_size, show_progress=True
        )
    elif task_type == TaskType.SEQUENCE_GENERATION:
        runner = SequenceGenerationRunner(
            task_config, plm, batch_size=batch_size, show_progress=True
        )
    elif task_type == TaskType.PPI_CLASSIFICATION:
        runner = PPIClassificationRunner(
            task_config, plm, batch_size=batch_size, show_progress=True
        )
    else:
        console.print(f"[red]Unsupported task type: {task_type}[/red]")
        console.print(f"[yellow]Supported types: multilabel_classification, multiclass_classification, "
                     f"binary_classification, regression, token_classification, text_generation, "
                     f"sequence_generation, ppi_classification[/yellow]")
        raise typer.Exit(1)

    # Run evaluation
    console.print("[yellow]Running evaluation...[/yellow]")
    console.print()

    try:
        results = runner.run(df)
    except Exception as e:
        console.print(f"[red]Error during evaluation: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)

    # Display results
    console.print()
    console.print("[bold green]Results:[/bold green]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Score", justify="right")

    for metric, score in results.items():
        table.add_row(metric, f"{score:.4f}")

    console.print(table)

    # Save results
    output_data = {
        "model": model,
        "task": task,
        "metrics": results,
        "config": {
            "batch_size": batch_size,
            "max_samples": max_samples,
            "data_path": str(data_file),
        },
    }

    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print()
    console.print(f"[green]Results saved to {output}[/green]")


@app.command("list-models")
def list_models():
    """List all available protein language models."""
    from procap.models.registry import ModelRegistry

    registry = ModelRegistry()

    console.print("[bold blue]Available Models[/bold blue]")
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Batch Size")
    table.add_column("Max Length")

    for name in registry.list_models():
        config = registry.get_config(name)
        params = config.get("params", {})
        model_class = config.get("class")

        model_type = "unknown"
        if model_class:
            model_type = model_class.__name__.replace("Adapter", "")

        table.add_row(
            name,
            model_type,
            str(params.get("max_batch_size", "-")),
            str(params.get("max_sequence_length", "-")),
        )

    console.print(table)

    console.print()
    console.print("[dim]Use 'procap info --model <name>' for details[/dim]")


@app.command("list-tasks")
def list_tasks(
    bloom_level: Optional[str] = typer.Option(
        None, "--bloom", "-b", help="Filter by Bloom level"
    ),
):
    """List all available benchmark tasks."""
    from procap.tasks.registry import _global_registry as registry
    from procap.tasks.schemas import BloomLevel

    console.print("[bold blue]Available Tasks[/bold blue]")
    console.print()

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Bloom Level")
    table.add_column("Type")
    table.add_column("Metrics")

    tasks = registry.get_all()

    if bloom_level:
        try:
            level = BloomLevel(bloom_level)
            tasks = [t for t in tasks if t.bloom_level == level]
        except ValueError:
            console.print(f"[red]Invalid Bloom level: {bloom_level}[/red]")
            console.print(f"Valid levels: {[l.value for l in BloomLevel]}")
            raise typer.Exit(1)

    for task in tasks:
        table.add_row(
            task.name,
            task.bloom_level.value if hasattr(task.bloom_level, 'value') else str(task.bloom_level),
            task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type),
            ", ".join(task.metrics[:3]) + ("..." if len(task.metrics) > 3 else ""),
        )

    console.print(table)


@app.command()
def info(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Task name"),
):
    """Get detailed information about a model or task."""
    if model:
        from procap.models.registry import ModelRegistry

        registry = ModelRegistry()
        try:
            config = registry.get_config(model)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

        console.print(f"[bold blue]Model: {model}[/bold blue]")
        console.print()

        params = config.get("params", {})
        for key, value in params.items():
            console.print(f"  {key}: {value}")

    if task:
        from procap.tasks.registry import get_task

        try:
            task_config = get_task(task)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

        console.print(f"[bold blue]Task: {task_config.name}[/bold blue]")
        console.print()
        console.print(f"  Bloom Level: {task_config.bloom_level}")
        console.print(f"  Domain: {task_config.domain}")
        console.print(f"  Type: {task_config.task_type}")
        console.print(f"  Description: {task_config.description}")
        console.print(f"  Metrics: {', '.join(task_config.metrics)}")
        console.print(f"  Dataset: {task_config.dataset_path}")
        console.print(f"  Input fields: {task_config.input_fields}")
        console.print(f"  Target: {task_config.target_field}")

    if not model and not task:
        console.print("[yellow]Specify --model or --task to get info[/yellow]")


@app.command()
def test(
    model: str = typer.Option("esm2_t6_8M", "--model", "-m", help="Model to test"),
):
    """Run a quick test to verify model works correctly."""
    from procap.models.registry import get_model

    console.print(f"[bold blue]Testing model: {model}[/bold blue]")
    console.print()

    # Test sequences
    test_sequences = [
        "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGP",
        "MLSRVLNRAEWLVSRRQICLSSVR",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    ]

    console.print("[yellow]Loading model...[/yellow]")
    try:
        plm = get_model(model)
        plm.load()
        console.print(f"[green]Model loaded successfully[/green]")
        console.print(f"  Hidden size: {plm.hidden_size}")
        console.print(f"  Device: {plm.device}")
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print("[yellow]Testing tokenization...[/yellow]")
    try:
        tokens = plm.tokenize(test_sequences)
        console.print(f"[green]Tokenization successful[/green]")
        console.print(f"  Input shape: {tokens.input_ids.shape}")
    except Exception as e:
        console.print(f"[red]Tokenization failed: {e}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print("[yellow]Testing embedding extraction...[/yellow]")
    try:
        embeddings = plm.get_embeddings(test_sequences, pooling="mean")
        console.print(f"[green]Embedding extraction successful[/green]")
        console.print(f"  Embedding shape: {embeddings.shape}")
        console.print(f"  Expected: [{len(test_sequences)}, {plm.hidden_size}]")
    except Exception as e:
        console.print(f"[red]Embedding extraction failed: {e}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print("[bold green]All tests passed![/bold green]")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
